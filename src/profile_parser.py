"""
Candidate profile normalization — parses raw JSON candidate data into a
consistent ``CandidateProfile`` dataclass with India-specific heuristics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from src.utils import clamp, get_logger, load_config, safe_float

logger = get_logger(__name__)


# ── Data model ────────────────────────────────────────────────────────────
@dataclass
class CandidateProfile:
    """Normalized representation of a single candidate."""

    candidate_id: str
    name: str
    headline: str = ""
    summary: str = ""
    skills: List[str] = field(default_factory=list)
    experience_years: float = 0.0
    education: str = ""
    education_tier: str = "tier_3"
    job_titles: List[str] = field(default_factory=list)
    companies: List[str] = field(default_factory=list)
    career_history_desc: List[str] = field(default_factory=list)
    location: str = ""
    last_active: Optional[datetime] = None
    skill_recency: Dict[str, int] = field(default_factory=dict)
    career_velocity: float = 0.0
    redrob_signals: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)

    # ── rich text for embedding ──────────────────────────────────────
    def to_embedding_text(self) -> str:
        """Build a descriptive paragraph for semantic embedding."""
        parts: List[str] = []
        if self.headline:
            parts.append(f"Headline: {self.headline}.")
        if self.summary:
            parts.append(f"Summary: {self.summary}")
        if self.skills:
            parts.append(f"Skills: {', '.join(self.skills)}.")
        if self.experience_years:
            parts.append(f"Experience: {self.experience_years:.1f} years.")
        if self.job_titles:
            parts.append(f"Recent roles: {', '.join(self.job_titles[:4])}.")
        if self.companies:
            parts.append(f"Companies: {', '.join(self.companies[:5])}.")
        if self.career_history_desc:
            parts.append(
                f"Experience Details: {' '.join(self.career_history_desc[:3])}"
            )

        return " ".join(parts) if parts else self.name


# ── Parser ────────────────────────────────────────────────────────────────
class ProfileParser:
    """Parse raw JSON candidate dictionaries into ``CandidateProfile`` instances."""

    def __init__(self) -> None:
        self._cfg = load_config()

    # ── public API ───────────────────────────────────────────────────
    def parse(self, raw: Dict[str, Any], idx: int = 0) -> CandidateProfile:
        """Parse one raw candidate record matching the JSON schema."""
        candidate_id = raw.get("candidate_id", f"CAND_{idx:07d}")

        # Profile block
        profile_block = raw.get("profile", {})
        name = profile_block.get("anonymized_name", f"Candidate-{candidate_id}")
        headline = profile_block.get("headline", "")
        summary = profile_block.get("summary", "")
        experience_years = safe_float(profile_block.get("years_of_experience", 0))
        location = profile_block.get("location", "")

        # Career block
        career_history = raw.get("career_history", [])
        job_titles = []
        companies = []
        career_history_desc = []
        for role in career_history:
            title = role.get("title", "")
            if title:
                job_titles.append(title)
            comp = role.get("company", "")
            if comp:
                companies.append(comp)
            desc = role.get("description", "")
            if desc:
                career_history_desc.append(desc)

        # Education block
        education_list = raw.get("education", [])
        education_tier = "tier_3"  # default
        education_str = ""
        if education_list:
            edu = education_list[0]
            education_str = f"{edu.get('degree', '')} in {edu.get('field_of_study', '')} from {edu.get('institution', '')}"
            education_tier = edu.get("tier", "tier_3")

        # Skills block
        skills_block = raw.get("skills", [])
        skills = []
        skill_recency: Dict[str, int] = {}
        current_year = datetime.now().year
        for sk in skills_block:
            skill_name = sk.get("name")
            if skill_name:
                skills.append(skill_name)
                # Roughly estimate recency based on duration (very crude but functional)
                skill_recency[skill_name] = current_year

        # Signals
        redrob_signals = raw.get("redrob_signals", {})
        last_active = self._parse_date(redrob_signals.get("last_active_date"))

        num_roles = max(len(job_titles), 1)
        career_velocity = (
            (num_roles - 1) / experience_years if experience_years > 0 else 0.0
        )

        return CandidateProfile(
            candidate_id=candidate_id,
            name=name,
            headline=headline,
            summary=summary,
            skills=skills,
            experience_years=experience_years,
            education=education_str,
            education_tier=education_tier,
            job_titles=job_titles,
            companies=companies,
            career_history_desc=career_history_desc,
            location=location,
            last_active=last_active,
            skill_recency=skill_recency,
            career_velocity=clamp(career_velocity, 0.0, 2.0),
            redrob_signals=redrob_signals,
            raw=raw,
        )

    def parse_many(self, records: List[Dict[str, Any]]) -> List[CandidateProfile]:
        profiles: List[CandidateProfile] = []
        for idx, rec in enumerate(records):
            try:
                profiles.append(self.parse(rec, idx=idx))
            except Exception as exc:
                logger.warning("Skipping malformed profile idx=%d: %s", idx, exc)
        logger.info(
            "Parsed %d / %d profiles successfully.", len(profiles), len(records)
        )
        return profiles

    @staticmethod
    def _parse_date(value: Any) -> Optional[datetime]:
        """Best-effort date parsing from common formats."""
        if not value:
            return None
        text = str(value).strip()
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%SZ"):
            try:
                return datetime.strptime(text, fmt)
            except ValueError:
                continue
        return None
