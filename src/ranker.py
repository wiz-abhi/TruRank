"""
Ranking engine — orchestrates the full pipeline from raw data to
ranked CSV output.

Usage:
    python -m src.ranker --jd data/sample/jd.json --profiles data/sample/profiles.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.embeddings import EmbeddingEngine
from src.explainer import generate_flags, generate_reasons
from src.jd_parser import JDParser, JobDescription
from src.profile_parser import CandidateProfile, ProfileParser
from src.signals import SignalComputer, SignalScores
from src.utils import OUTPUTS_DIR, ensure_dir, get_logger, load_config

logger = get_logger(__name__)


# ── Ranked candidate dataclass ───────────────────────────────────────────
@dataclass
class RankedCandidate:
    """A candidate with all scoring attached."""

    rank: int
    candidate_id: str
    name: str
    composite_score: float
    match_percentage: int
    semantic_score: float
    skill_match_score: float
    experience_years: float
    signal_scores: Dict[str, float] = field(default_factory=dict)
    top_3_reasons: List[str] = field(default_factory=list)
    flag_notes: str = ""


# ── Ranker ───────────────────────────────────────────────────────────────
class CandidateRanker:
    """End-to-end ranking pipeline.

    Orchestrates: JD parsing → profile parsing → embedding → signal
    computation → sorting → explanation → CSV export.
    """

    def __init__(self, embedding_engine: Optional[EmbeddingEngine] = None) -> None:
        """Initialise with shared components.

        Args:
            embedding_engine: Reuse an existing engine (e.g., from the app).
        """
        self._jd_parser = JDParser()
        self._profile_parser = ProfileParser()
        self._embedder = embedding_engine or EmbeddingEngine()
        self._signal_computer = SignalComputer()
        self._cfg = load_config()

    # ── public API ───────────────────────────────────────────────────
    def rank(
        self,
        profiles: List[CandidateProfile],
        jd: JobDescription,
        refresh_embeddings: bool = False,
    ) -> List[RankedCandidate]:
        """Rank all profiles against the JD.

        Args:
            profiles: List of normalised profiles.
            jd: Parsed job description.
            refresh_embeddings: If True, ignore cached embeddings.

        Returns:
            Sorted list of ``RankedCandidate`` (best first).
        """
        if not profiles:
            logger.warning("No profiles to rank.")
            return []

        # 1) Embed JD
        logger.info("Embedding JD …")
        jd_vec = self._embedder.embed_jd(jd)

        # 2) Batch-embed profiles
        logger.info("Embedding %d profiles …", len(profiles))
        profile_vecs = self._embedder.batch_embed_profiles(
            profiles,
            cache_key="profiles_batch",
            refresh=refresh_embeddings,
        )

        # 3) Compute semantic similarities in bulk
        similarities = self._embedder.batch_similarity(jd_vec, profile_vecs)

        # 4) Compute all signals
        logger.info("Computing signals for %d candidates …", len(profiles))
        candidates: List[RankedCandidate] = []

        for i, profile in enumerate(profiles):
            sem_sim = float(similarities[i])
            scores: SignalScores = self._signal_computer.compute_all(
                profile, jd, sem_sim
            )
            reasons = generate_reasons(profile, jd, scores, rank=0)
            flags = generate_flags(profile, jd, scores)

            rc = RankedCandidate(
                rank=0,  # assigned after sorting
                candidate_id=profile.candidate_id,
                name=profile.name,
                composite_score=round(scores.composite_score, 4),
                match_percentage=int(round(scores.composite_score * 100)),
                semantic_score=round(scores.semantic_similarity, 4),
                skill_match_score=round(scores.skill_match, 4),
                experience_years=profile.experience_years,
                signal_scores=scores.to_dict(),
                top_3_reasons=reasons,
                flag_notes=flags,
            )
            candidates.append(rc)

        # 5) Sort by composite score (descending)
        candidates.sort(key=lambda c: c.composite_score, reverse=True)

        # 6) Assign ranks
        for idx, c in enumerate(candidates, start=1):
            c.rank = idx

        logger.info(
            "Ranking complete — top candidate: %s (%.2f%%)",
            candidates[0].name if candidates else "N/A",
            candidates[0].match_percentage if candidates else 0,
        )
        return candidates

    def rank_from_raw(
        self,
        raw_profiles: List[Dict[str, Any]],
        jd_source: str | Dict[str, Any],
        refresh: bool = False,
    ) -> List[RankedCandidate]:
        """Convenience: parse raw data and rank in one call.

        Args:
            raw_profiles: List of raw profile dicts.
            jd_source: Raw JD text or dict.
            refresh: Refresh embeddings cache.

        Returns:
            Sorted ``RankedCandidate`` list.
        """
        jd = self._jd_parser.parse(jd_source)
        profiles = self._profile_parser.parse_many(raw_profiles)
        return self.rank(profiles, jd, refresh_embeddings=refresh)

    # ── CSV export ───────────────────────────────────────────────────
    @staticmethod
    def export_csv(ranked: List[RankedCandidate], path: str | Path) -> Path:
        """Write ranked results to CSV in the required submission format.

        Args:
            ranked: Sorted list of ranked candidates.
            path: Output file path.

        Returns:
            Resolved path of the written file.
        """
        out = Path(path)
        ensure_dir(out.parent)

        rows: List[Dict[str, Any]] = []
        for c in ranked:
            reasons = c.top_3_reasons + [""] * 3  # pad
            rows.append(
                {
                    "rank": c.rank,
                    "candidate_id": c.candidate_id,
                    "name": c.name,
                    "match_percentage": c.match_percentage,
                    "composite_score": c.composite_score,
                    "semantic_score": c.semantic_score,
                    "skill_match_score": c.skill_match_score,
                    "experience_years": c.experience_years,
                    "top_reason_1": reasons[0],
                    "top_reason_2": reasons[1],
                    "top_reason_3": reasons[2],
                    "flag_notes": c.flag_notes,
                }
            )

        df = pd.DataFrame(rows)
        columns = (
            load_config().get("output", {}).get("csv_columns", list(rows[0].keys()))
        )
        df = df[columns]
        df.to_csv(out, index=False)
        logger.info("Exported %d ranked candidates to %s", len(ranked), out)
        return out.resolve()


# ── CLI entrypoint ───────────────────────────────────────────────────────
def _load_profiles_file(path: Path) -> List[Dict[str, Any]]:
    """Load profiles from CSV, JSON, or XLSX."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = pd.DataFrame([data])
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")

    logger.info("Loaded %d rows from %s", len(df), path)
    return df.to_dict(orient="records")


def _load_jd(path: Path) -> str | Dict[str, Any]:
    """Load JD from text, JSON, or TXT file."""
    suffix = path.suffix.lower()
    with open(path, "r", encoding="utf-8") as f:
        if suffix == ".json":
            return json.load(f)
        return f.read()


def main() -> None:
    """CLI entrypoint for the ranking pipeline."""
    parser = argparse.ArgumentParser(
        description="Rank candidates against a Job Description."
    )
    parser.add_argument("--jd", required=True, help="Path to JD file (JSON or TXT)")
    parser.add_argument(
        "--profiles", required=True, help="Path to profiles file (CSV/JSON/XLSX)"
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUTS_DIR / "ranked_output.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--refresh", action="store_true", help="Refresh embedding cache"
    )
    parser.add_argument(
        "--top", type=int, default=0, help="Only output top N candidates"
    )

    args = parser.parse_args()

    jd_source = _load_jd(Path(args.jd))
    raw_profiles = _load_profiles_file(Path(args.profiles))

    ranker = CandidateRanker()
    ranked = ranker.rank_from_raw(raw_profiles, jd_source, refresh=args.refresh)

    if args.top > 0:
        ranked = ranked[: args.top]

    CandidateRanker.export_csv(ranked, args.output)
    print(f"\n✅ Ranked {len(ranked)} candidates → {args.output}")


if __name__ == "__main__":
    main()
