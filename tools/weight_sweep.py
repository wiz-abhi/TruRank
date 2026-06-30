#!/usr/bin/env python3
"""
Silver-label weight optimization.

Step 1: Build a ~200-candidate graded silver-label set by scoring a stratified
        sample with a "teacher" scorer that uses generous thresholds.
Step 2: Grid-sweep the 13 signal weights to maximize NDCG@10 on that set.
Step 3: Print the optimal weights for pasting into config.yaml / signals.py.

This runs offline — not part of the timed pipeline.
"""

import itertools
import json
import math
import pickle
import random
import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.jd_parser import JobDescription
from src.profile_parser import CandidateProfile
from src.signals import SignalComputer, SignalScores, clamp
from src.utils import load_config


# ═══════════════════════════════════════════════════════════════════════════
# SILVER LABEL GENERATION
# ═══════════════════════════════════════════════════════════════════════════

def _teacher_relevance(profile: CandidateProfile, jd, semantic_sim: float) -> float:
    """
    Generate a graded relevance label (0–4) for a candidate.
    
    This is a "teacher" scorer that uses generous, interpretable rules
    rather than the composite score we're trying to optimize. The idea:
    if we train the weights to match THIS teacher, the composite score
    will be well-calibrated.
    
    Grades:
        4 = Perfect fit (retrieval/ranking role at product co, 5–9y, India, active)
        3 = Strong fit (relevant career, most criteria met)
        2 = Moderate fit (some relevant experience)
        1 = Weak fit (tangentially related)
        0 = Not relevant
    """
    grade = 0.0
    
    # Career text substance (strongest signal)
    text = " ".join(profile.career_history_desc).lower()
    current_title = (profile.raw.get("profile", {}).get("current_title", "") or "").lower()
    
    retrieval_terms = ["retrieval", "ranking", "search", "recommendation", "recommender", "vector", "semantic"]
    production_terms = ["production", "deployed", "shipped", "scale", "real-time", "pipeline", "users"]
    eval_terms = ["ndcg", "mrr", "map", "a/b", "evaluation", "benchmark"]
    
    ret_count = sum(1 for t in retrieval_terms if t in text)
    prod_count = sum(1 for t in production_terms if t in text)
    eval_count = sum(1 for t in eval_terms if t in text)
    
    if ret_count >= 3 and prod_count >= 2:
        grade += 2.0
    elif ret_count >= 2 or prod_count >= 2:
        grade += 1.0
    elif ret_count >= 1:
        grade += 0.5
    
    if eval_count >= 2:
        grade += 0.5
    
    # Title relevance
    relevant_titles = ["machine learning", "ml", "ai", "nlp", "search", "recommend", "data scientist", "applied scientist"]
    if any(t in current_title for t in relevant_titles):
        grade += 0.5
    
    # Experience band (5–9 is ideal)
    yrs = profile.experience_years
    if 5 <= yrs <= 9:
        grade += 0.5
    elif 3 <= yrs <= 12:
        grade += 0.25
    
    # Location
    loc = profile.location.lower()
    if any(c in loc for c in ("pune", "noida")):
        grade += 0.3
    elif any(c in loc for c in ("hyderabad", "mumbai", "delhi", "bangalore", "bengaluru")):
        grade += 0.15
    
    # Product company background
    roles = profile.raw.get("career_history", [])
    product_terms = ("software product", "internet", "e-commerce", "fintech", "saas", "marketplace")
    if any(any(pt in str(r.get("industry", "")).lower() for pt in product_terms) for r in roles if isinstance(r, dict)):
        grade += 0.3
    
    # Semantic similarity as a soft signal
    if semantic_sim > 0.6:
        grade += 0.4
    elif semantic_sim > 0.4:
        grade += 0.2
    
    return min(4.0, grade)


def build_silver_labels(cache_path: str, n_sample: int = 200) -> list:
    """
    Build a stratified silver-label set.
    
    Samples candidates from different semantic similarity buckets to ensure
    the set covers the full quality spectrum (not just the head).
    """
    from sentence_transformers import SentenceTransformer
    from src.profile_parser import CandidateProfile
    
    print(f"Loading cache from {cache_path}...")
    with open(cache_path, "rb") as f:
        cache_data = pickle.load(f)
    
    profiles = cache_data["profiles"]
    embeddings = cache_data["embeddings"]
    model_name = cache_data["model"]
    
    cfg = load_config()
    target_jd_cfg = cfg.get("target_jd", {})
    jd = JobDescription(
        raw_text="Target Hackathon JD",
        required_skills=target_jd_cfg.get("required_skills", []),
        preferred_skills=target_jd_cfg.get("preferred_skills", []),
        min_experience_years=target_jd_cfg.get("min_experience", 4),
        domain=target_jd_cfg.get("domain", "data_science"),
        culture_signals=target_jd_cfg.get("culture_signals", []),
    )
    jd_embedding_text = """
Senior AI/ML Engineer — Production Retrieval, Ranking and Recommendation Systems.
Seeking engineers who have shipped embedding-based retrieval, semantic search, or
recommendation systems to real users at scale.
"""
    
    model = SentenceTransformer(model_name)
    jd_emb = model.encode(jd_embedding_text, normalize_embeddings=True)
    
    prof_norms = np.asarray(embeddings, dtype=np.float32)
    jd_norm = np.asarray(jd_emb, dtype=np.float32)
    sims = prof_norms @ jd_norm
    
    # Stratified sampling: 5 buckets by semantic similarity
    indices = list(range(len(profiles)))
    random.seed(42)
    
    buckets = [
        ("top_50", sorted(range(len(sims)), key=lambda i: -sims[i])[:50]),
        ("top_200", sorted(range(len(sims)), key=lambda i: -sims[i])[50:200]),
        ("mid_500", sorted(range(len(sims)), key=lambda i: -sims[i])[200:700]),
        ("tail_random", random.sample(range(700, len(profiles)), min(100, len(profiles) - 700))),
    ]
    
    sampled = set()
    per_bucket = n_sample // len(buckets)
    for name, bucket_indices in buckets:
        chosen = random.sample(bucket_indices, min(per_bucket, len(bucket_indices)))
        sampled.update(chosen)
        print(f"  Bucket '{name}': sampled {len(chosen)} candidates")
    
    # Build labels
    computer = SignalComputer()
    silver = []
    for idx in sampled:
        profile = profiles[idx]
        sim = float(sims[idx])
        scores = computer.compute_all(profile, jd, sim)
        relevance = _teacher_relevance(profile, jd, sim)
        silver.append({
            "idx": idx,
            "candidate_id": profile.candidate_id,
            "semantic_sim": sim,
            "relevance": relevance,
            "signals": scores.to_dict(),
        })
    
    print(f"Built {len(silver)} silver labels. Grade distribution:")
    from collections import Counter
    grade_dist = Counter(int(s["relevance"]) for s in silver)
    for g in sorted(grade_dist):
        print(f"  Grade {g}: {grade_dist[g]} candidates")
    
    return silver


# ═══════════════════════════════════════════════════════════════════════════
# NDCG COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════

def _dcg(relevances, k):
    """Discounted Cumulative Gain at k."""
    return sum(
        (2**rel - 1) / math.log2(i + 2)
        for i, rel in enumerate(relevances[:k])
    )


def ndcg_at_k(scores, relevances, k=10):
    """Compute NDCG@k given composite scores and relevance labels."""
    paired = sorted(zip(scores, relevances), key=lambda x: -x[0])
    predicted_rels = [r for _, r in paired]
    ideal_rels = sorted(relevances, reverse=True)
    
    dcg = _dcg(predicted_rels, k)
    idcg = _dcg(ideal_rels, k)
    
    return dcg / idcg if idcg > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# WEIGHT SWEEP
# ═══════════════════════════════════════════════════════════════════════════

SIGNAL_NAMES = [
    "semantic_similarity", "skill_match", "skill_evidence", "skill_corroboration",
    "career_evidence", "experience_fit", "domain_alignment", "culture_fit",
    "location_fit", "skill_recency", "career_stability", "product_company_fit",
    "work_mode_fit", "external_validation", "production_recency",
]


def score_with_weights(silver, weights):
    """Recompute composite scores using a given weight vector."""
    scores = []
    relevances = []
    for entry in silver:
        signals = entry["signals"]
        base = sum(w * signals.get(name, 0.0) for name, w in zip(SIGNAL_NAMES, weights))
        # Use the stored behavioral multiplier and trust
        behav = signals.get("behavioral_multiplier", 1.0)
        trust = signals.get("profile_trust", 1.0)
        composite = clamp(base * behav * trust, 0.0, 1.0)
        scores.append(composite)
        relevances.append(entry["relevance"])
    return scores, relevances


def sweep_weights(silver, n_random=5000):
    """
    Grid sweep + random search over weight space.
    
    We use a combination of:
    1. Perturbations around the current hand-tuned weights
    2. Purely random weight vectors (Dirichlet distribution)
    """
    # Current weights as baseline
    current = [0.15, 0.07, 0.10, 0.06, 0.28, 0.10, 0.07, 0.05, 0.05, 0.04, 0.06, 0.07, 0.02]
    
    best_ndcg10 = 0.0
    best_ndcg50 = 0.0
    best_weights = current
    best_combined = 0.0
    
    candidates = []
    
    # 1. Current baseline
    candidates.append(("baseline", current))
    
    # 2. Perturbations around baseline
    rng = random.Random(42)
    for i in range(2000):
        perturbed = [max(0.01, w + rng.gauss(0, 0.03)) for w in current]
        total = sum(perturbed)
        perturbed = [w / total for w in perturbed]  # normalize to sum=1
        candidates.append((f"perturb_{i}", perturbed))
    
    # 3. Random Dirichlet samples
    np_rng = np.random.RandomState(42)
    for i in range(n_random):
        # Use Dirichlet with higher alpha for career_evidence (index 4) to stay close to domain knowledge
        alpha = np.ones(len(SIGNAL_NAMES)) * 2.0
        alpha[4] = 8.0  # career_evidence should stay dominant
        alpha[0] = 5.0  # semantic_similarity important
        alpha[2] = 4.0  # skill_evidence
        alpha[5] = 4.0  # experience_fit
        
        weights = np_rng.dirichlet(alpha).tolist()
        candidates.append((f"dirichlet_{i}", weights))
    
    # 4. Evaluate all candidates
    print(f"Sweeping {len(candidates)} weight vectors...")
    for name, weights in candidates:
        scores, relevances = score_with_weights(silver, weights)
        n10 = ndcg_at_k(scores, relevances, k=10)
        n50 = ndcg_at_k(scores, relevances, k=50)
        
        # Combined metric matching the challenge scoring
        combined = 0.50 * n10 + 0.30 * n50 + 0.15 * ndcg_at_k(scores, relevances, k=len(scores)) + 0.05 * n10
        
        if combined > best_combined:
            best_combined = combined
            best_ndcg10 = n10
            best_ndcg50 = n50
            best_weights = weights
            if "baseline" not in name:
                print(f"  New best ({name}): NDCG@10={n10:.4f}, NDCG@50={n50:.4f}, Combined={combined:.4f}")
    
    return best_weights, best_ndcg10, best_ndcg50, best_combined


def main():
    cache_path = "data/processed/candidates_cache.pkl"
    
    if not Path(cache_path).exists():
        print(f"Cache not found at {cache_path}. Run precompute.py first.")
        sys.exit(1)
    
    print("=" * 70)
    print("SILVER LABEL GENERATION")
    print("=" * 70)
    silver = build_silver_labels(cache_path, n_sample=200)
    
    # Save silver labels
    silver_path = Path("data/processed/silver_labels.json")
    silver_path.parent.mkdir(parents=True, exist_ok=True)
    with open(silver_path, "w") as f:
        json.dump(silver, f, indent=2)
    print(f"\nSaved silver labels to {silver_path}")
    
    print("\n" + "=" * 70)
    print("WEIGHT OPTIMIZATION")
    print("=" * 70)
    
    # Baseline evaluation
    current_weights = [0.15, 0.07, 0.10, 0.06, 0.28, 0.10, 0.07, 0.05, 0.05, 0.04, 0.06, 0.07, 0.02]
    scores, relevances = score_with_weights(silver, current_weights)
    baseline_n10 = ndcg_at_k(scores, relevances, k=10)
    baseline_n50 = ndcg_at_k(scores, relevances, k=50)
    print(f"\nBaseline NDCG@10: {baseline_n10:.4f}")
    print(f"Baseline NDCG@50: {baseline_n50:.4f}")
    
    best_weights, best_n10, best_n50, best_combined = sweep_weights(silver)
    
    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")
    print(f"Baseline  NDCG@10: {baseline_n10:.4f}")
    print(f"Optimized NDCG@10: {best_n10:.4f} ({(best_n10 - baseline_n10) / max(baseline_n10, 0.001) * 100:+.1f}%)")
    print(f"Optimized NDCG@50: {best_n50:.4f}")
    print(f"Optimized Combined: {best_combined:.4f}")
    
    print(f"\nOptimal weights (paste into signals.py):")
    for name, w in zip(SIGNAL_NAMES, best_weights):
        print(f'    w.get("{name}", {w:.4f}) * scores.{name}')
    
    # Save optimal weights
    weights_path = Path("data/processed/optimal_weights.json")
    with open(weights_path, "w") as f:
        json.dump(dict(zip(SIGNAL_NAMES, [round(w, 4) for w in best_weights])), f, indent=2)
    print(f"\nSaved optimal weights to {weights_path}")


if __name__ == "__main__":
    main()
