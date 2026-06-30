import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import argparse
import csv
import pickle
import time
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from src.jd_parser import JobDescription
from src.profile_parser import CandidateProfile
from src.signals import SignalComputer
from src.explainer import ExplainerEngine
from src.honeypot_detector import HoneypotDetector
from src.utils import load_config


def run_ranking(cache_path: str, output_path: str):
    start_time = time.time()
    print("Loading configuration...")
    cfg = load_config()
    target_jd_cfg = cfg.get("target_jd", {})

    # Create the JD representation
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
recommendation systems to real users at scale. Must demonstrate: production deployment
of dense retrieval or hybrid search (Pinecone, Weaviate, Qdrant, Milvus, FAISS,
Elasticsearch); evaluation with NDCG, MRR, MAP, A/B testing, offline-to-online
correlation; handling embedding drift, index refresh, retrieval quality regression.
Strong Python. Product company background preferred. 5-9 years applied ML experience.
Location: Pune or Noida preferred. Immediate to 30-day notice. NLP/IR domain expertise.
LTR, fine-tuning (LoRA/QLoRA/PEFT), or distributed systems a strong plus.
"""

    print(f"Loading precomputed cache from {cache_path}...")
    with open(cache_path, "rb") as f:
        cache_data = pickle.load(f)

    profiles: List[CandidateProfile] = cache_data["profiles"]
    embeddings: np.ndarray = cache_data["embeddings"]
    model_name = cache_data["model"]

    print(
        f"Loaded {len(profiles)} candidates. Loading model '{model_name}' for JD embedding..."
    )
    model = SentenceTransformer(model_name)

    # ── Aspect-based JD embeddings ──────────────────────────────────────────────
    # Instead of one coarse JD embedding, we decompose the JD into 6 focused
    # aspect queries. Each retrieves independently; RRF fuses the rankings.
    # This surfaces "hidden gems" who are strong in NLP/IR but don't use buzzwords.
    JD_ASPECTS = {
        "retrieval_search": (
            "embeddings retrieval vector search dense FAISS Pinecone Weaviate Qdrant Milvus "
            "semantic search approximate nearest neighbor ANN index hybrid search"
        ),
        "ranking_eval": (
            "ranking reranking evaluation NDCG MRR MAP A/B testing offline-to-online "
            "learning to rank LTR cross-encoder bi-encoder retrieval quality"
        ),
        "nlp_ir": (
            "NLP natural language processing information retrieval language models transformers "
            "BERT sentence-transformers text understanding fine-tuning"
        ),
        "production_recency": (
            "production deployment shipped real users scale live system serving "
            "inference pipeline engineering MLOps"
        ),
        "product_company": (
            "product company SaaS startup internet e-commerce fintech shipped users "
            "ownership product thinking"
        ),
        "location_availability": (
            "India Pune Noida Bangalore immediate notice period available joining"
        ),
    }

    print("Computing aspect-based JD embeddings...")
    aspect_embeddings = {
        name: model.encode(text, normalize_embeddings=True)
        for name, text in JD_ASPECTS.items()
    }

    print("Computing per-aspect semantic similarities (vectorized)...")
    prof_norms = np.asarray(embeddings, dtype=np.float32)
    aspect_scores: dict = {}
    for name, asp_emb in aspect_embeddings.items():
        asp_norm = np.asarray(asp_emb, dtype=np.float32)
        aspect_scores[name] = prof_norms @ asp_norm

    # Weighted aggregate semantic score for signal computation
    ASPECT_WEIGHTS = {
        "retrieval_search": 0.30,
        "ranking_eval":     0.25,
        "nlp_ir":           0.20,
        "production_recency": 0.10,
        "product_company":  0.08,
        "location_availability": 0.07,
    }
    semantic_scores = sum(
        ASPECT_WEIGHTS[name] * scores
        for name, scores in aspect_scores.items()
    )

    # ── BM25 lexical retrieval ──────────────────────────────────────────────────
    from rank_bm25 import BM25Okapi

    BM25_QUERY_TERMS = [
        "retrieval", "ranking", "embedding", "semantic", "search", "recommendation",
        "vector", "production", "deployed", "ndcg", "mrr", "evaluation", "faiss",
        "pinecone", "qdrant", "weaviate", "elasticsearch", "pipeline", "scale",
        "reranking", "ltr", "cross-encoder", "bi-encoder", "information retrieval",
    ]

    def get_description_text(profile):
        return " ".join(profile.career_history_desc).lower()

    print("Building BM25 index over career descriptions...")
    corpus = [get_description_text(p).split() for p in profiles]
    bm25 = BM25Okapi(corpus)
    bm25_scores_raw = np.array(bm25.get_scores(BM25_QUERY_TERMS), dtype=np.float32)

    # ── Reciprocal Rank Fusion (8 independent rankings) ──────────────────────
    # RRF is scale-free: only rank positions matter, avoiding normalization hacks.
    # score(id) = Σ 1/(k + rank_R(id)) over each ranking R
    #
    # KEY IMPROVEMENT: Instead of RRF-ing just 2 rankings (aggregate_semantic, BM25),
    # we RRF ALL 8 rankings independently:
    #   6 per-aspect semantic rankings + 1 BM25 + 1 aggregate semantic
    # This gives RRF 8 independent views — a candidate strong in one aspect
    # (e.g., NLP/IR but not embeddings) still surfaces even if their aggregate
    # semantic score is mediocre.
    SHORTLIST_SIZE = 800
    RRF_K = 60

    print(f"Fusing {len(aspect_scores) + 2} rankings with RRF (k={RRF_K}), shortlist={SHORTLIST_SIZE}...")
    n = len(profiles)
    rrf_scores = np.zeros(n, dtype=np.float64)

    # Per-aspect semantic rankings (6 independent views)
    for name, a_scores in aspect_scores.items():
        order = np.argsort(-a_scores)
        for rank, idx in enumerate(order):
            rrf_scores[idx] += 1.0 / (RRF_K + rank + 1)

    # Aggregate semantic ranking (weighted combination)
    sem_order = np.argsort(-semantic_scores)
    for rank, idx in enumerate(sem_order):
        rrf_scores[idx] += 1.0 / (RRF_K + rank + 1)

    # BM25 lexical ranking
    bm25_order = np.argsort(-bm25_scores_raw)
    for rank, idx in enumerate(bm25_order):
        rrf_scores[idx] += 1.0 / (RRF_K + rank + 1)

    top_k_indices = np.argpartition(-rrf_scores, SHORTLIST_SIZE - 1)[:SHORTLIST_SIZE]
    print(f"RRF shortlist size: {len(top_k_indices)} candidates")

    # ── Full 12-signal scoring on the shortlist ─────────────────────────────────
    print("Computing behavioral signals and final composites for shortlist...")
    computer = SignalComputer()
    scored_candidates = []

    for i in top_k_indices:
        profile = profiles[i]
        sim = float(semantic_scores[i])
        scores = computer.compute_all(profile, jd, sim)
        scored_candidates.append((scores.composite_score, profile.candidate_id, profile, scores))

    # Sort: Primary by score (descending), Secondary by candidate_id (ascending)
    print("Sorting candidates...")
    scored_candidates.sort(key=lambda x: (-x[0], x[1]))

    # ── Cross-encoder rerank (head of the shortlist) ────────────────────────────
    # Rerank the top 200 by running full cross-attention over (JD, candidate) pairs.
    # This separates real retrieval/ranking career substance from keyword stuffers.
    from src.cross_encoder import rerank as ce_rerank
    print("Running cross-encoder rerank on top 200...")
    scored_candidates = ce_rerank(jd_embedding_text, scored_candidates, top_n=200, ce_weight=0.10)


    print("Running honeypot detection on top candidates...")
    detector = HoneypotDetector()
    top_100 = []
    honeypots_caught = 0
    
    # Keep pulling until we have 100 safe candidates
    for score, cid, profile, scores in scored_candidates:
        if len(top_100) >= 100:
            break
            
        # Detect honeypots
        hp_result = detector.detect(profile.raw)
        if hp_result.is_honeypot:
            honeypots_caught += 1
            continue
            
        top_100.append((score, cid, profile, scores))
        
    print(f"Caught and filtered {honeypots_caught} honeypots from the top ranking.")

    print("Generating explanations for top 100...")
    explainer = ExplainerEngine()
    results = []

    for rank, (score, cid, profile, scores) in enumerate(top_100, start=1):
        # Generate reasoning based on the new SignalScores structure
        reasoning = explainer.explain_rank(profile, scores, jd, rank=rank, total=len(top_100))
        # Format the reasoning properly to fit in CSV (no newlines in the cell)
        reasoning = reasoning.replace("\n", " ").replace("\r", "").strip()
        results.append(
            {
                "candidate_id": cid,
                "rank": rank,
                "score": f"{score:.6f}",
                "reasoning": reasoning,
            }
        )

    print(f"Writing output to {output_path}...")
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["candidate_id", "rank", "score", "reasoning"]
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"Done in {time.time() - start_time:.2f}s total.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rank candidates from cache")
    parser.add_argument(
        "--candidates",
        type=str,
        default="data/raw/candidates.jsonl",
        help="Input candidates file (ignored in cached mode)",
    )
    parser.add_argument(
        "--out", type=str, default="submission.csv", help="Output CSV path"
    )
    parser.add_argument(
        "--cache", type=str, default="data/processed/candidates_cache.pkl", help="Input cache file"
    )
    args = parser.parse_args()

    run_ranking(args.cache, args.out)
