import argparse
import csv
import pickle
import time
from typing import List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from src.jd_parser import JobDescription
from src.profile_parser import CandidateProfile
from src.signals import SignalComputer
from src.explainer import ExplainerEngine
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
    jd_embedding_text = (
        "Skills: "
        + ", ".join(jd.required_skills)
        + " Experience: "
        + str(jd.min_experience_years)
        + " years."
    )

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
    jd_embedding = model.encode(jd_embedding_text, normalize_embeddings=True)

    print("Computing semantic similarities (vectorized)...")
    # embeddings and jd_embedding are both normalized
    jd_norm = np.asarray(jd_embedding, dtype=np.float32)
    prof_norms = np.asarray(embeddings, dtype=np.float32)
    semantic_scores = prof_norms @ jd_norm

    print("Computing behavioral signals and final composites...")
    computer = SignalComputer()
    scored_candidates = []

    for i, profile in enumerate(profiles):
        sim = float(semantic_scores[i])
        scores = computer.compute_all(profile, jd, sim)
        # Round to 4 decimal places so the sort order matches what validate_submission.py sees
        rounded_score = round(scores.composite_score, 4)
        scored_candidates.append((rounded_score, profile.candidate_id, profile, scores))

    # Sort: Primary by score (descending), Secondary by candidate_id (ascending)
    print("Sorting candidates...")
    scored_candidates.sort(key=lambda x: (-x[0], x[1]))

    top_100 = scored_candidates[:100]

    print("Generating explanations for top 100...")
    explainer = ExplainerEngine()
    results = []

    for rank, (score, cid, profile, scores) in enumerate(top_100, start=1):
        # Generate reasoning based on the new SignalScores structure
        reasoning = explainer.explain_rank(profile, scores)
        # Format the reasoning properly to fit in CSV (no newlines in the cell)
        reasoning = reasoning.replace("\n", " ").replace("\r", "").strip()
        results.append(
            {
                "candidate_id": cid,
                "rank": rank,
                "score": f"{score:.4f}",
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
    args = parser.parse_args()

    cache_path = "data/processed/candidates_cache.pkl"
    run_ranking(cache_path, args.out)
