"""
Candidate profile normalization — parses raw JSON candidate data into a
consistent ``CandidateProfile`` dataclass with India-specific heuristics.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

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
        """Parse one raw candidate record matching the JSON schema or flat CSV."""
        candidate_id = raw.get("candidate_id", f"CAND_{idx:07d}")

        # Profile block (nested JSON vs flat CSV)
        profile_block = raw.get("profile")
        if isinstance(profile_block, str):
            # It's a flat CSV but happened to have 'profile' string, ignore it for parsing fields
            profile_block = {}
        elif not profile_block:
            profile_block = raw

        name = profile_block.get("anonymized_name") or raw.get("name") or f"Candidate-{candidate_id}"
        headline = profile_block.get("headline", "")
        summary = profile_block.get("summary", "")
        exp_val = profile_block.get("years_of_experience") or raw.get("experience_years") or 0
        experience_years = safe_float(exp_val)
        location = profile_block.get("location") or raw.get("location") or ""

        # Career block
        career_history = raw.get("career_history", [])
        job_titles = []
        companies = []
        career_history_desc = []
        
        if isinstance(career_history, str):
            # Flat CSV: job_titles, companies as comma separated strings
            titles_str = raw.get("job_titles", "")
            if pd.notna(titles_str) and titles_str:
                job_titles = [t.strip() for t in str(titles_str).split(",")]
            comps_str = raw.get("companies", "")
            if pd.notna(comps_str) and comps_str:
                companies = [c.strip() for c in str(comps_str).split(",")]
        elif isinstance(career_history, list):
            # Nested JSON
            for role in career_history:
                if isinstance(role, dict):
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
        education_val = raw.get("education", [])
        education_tier = "tier_3"  # default
        education_str = ""
        
        if isinstance(education_val, str):
            # Flat CSV
            education_str = education_val
        elif isinstance(education_val, list) and education_val:
            # Nested JSON
            edu = education_val[0]
            if isinstance(edu, dict):
                education_str = f"{edu.get('degree', '')} in {edu.get('field_of_study', '')} from {edu.get('institution', '')}"
                education_tier = edu.get("tier", "tier_3")

        # Skills block
        skills_val = raw.get("skills", [])
        skills = []
        skill_recency: Dict[str, int] = {}
        current_year = datetime.now().year
        
        if isinstance(skills_val, str):
            # Flat CSV
            if pd.notna(skills_val) and skills_val:
                skills = [s.strip() for s in skills_val.split(",")]
                for sk in skills:
                    skill_recency[sk] = current_year
        elif isinstance(skills_val, list):
            # Nested JSON
            for sk in skills_val:
                if isinstance(sk, dict):
                    skill_name = sk.get("name")
                    if skill_name:
                        skills.append(skill_name)
                        skill_recency[skill_name] = current_year

        # Signals
        redrob_signals = raw.get("redrob_signals", {})
        if isinstance(redrob_signals, str):
            redrob_signals = {}
            
        # Get last active date either from redrob signals or flat CSV root
        last_active_raw = redrob_signals.get("last_active_date") or raw.get("last_active")
        last_active = self._parse_date(last_active_raw)

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
