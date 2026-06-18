"""
Explainability engine — generates human-readable reasons for each
candidate's ranking, using template-based generation.
"""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from src.jd_parser import JobDescription
from src.profile_parser import CandidateProfile
from src.signals import SignalScores, _canonicalize_skill
from src.utils import get_logger

logger = get_logger(__name__)


class ExplainerEngine:
    def explain_rank(self, profile: CandidateProfile, scores: SignalScores) -> str:
        """Generate a single cohesive paragraph explaining the candidate's rank."""
        reasons = []

        # 1. Semantic + Experience
        sem = scores.semantic_similarity
        exp = profile.experience_years

        if sem >= 0.70:
            reasons.append(
                f"Strong semantic profile match ({sem:.0%}) with {exp:.1f} years of relevant experience."
            )
        elif sem >= 0.50:
            reasons.append(
                f"Moderate semantic match ({sem:.0%}) with {exp:.1f} years of experience."
            )
        else:
            reasons.append(
                f"Low semantic match ({sem:.0%}) but has {exp:.1f} years of experience."
            )

        # 2. Behavioral Signals (Redrob specific)
        behav = scores.behavioral_multiplier
        if behav > 1.2:
            reasons.append("Highly active and responsive candidate.")
        elif behav < 0.8:
            reasons.append(
                "Penalty applied due to low recruiter response rate or stale activity."
            )

        # 3. Domain / Culture
        dom = scores.domain_alignment
        if dom > 0.6:
            reasons.append(
                "Background aligns well with data science and startup domains."
            )

        # 4. Education Bonus
        if scores.education_tier_bonus > 0.1:
            reasons.append("Tier-1 educational background.")

        # Combine into a single string
        return " ".join(reasons)
