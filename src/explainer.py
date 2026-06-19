"""
Explainability engine — generates human-readable reasons for each
candidate's ranking, using candidate-specific profile details.
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
    def explain_rank(self, profile: CandidateProfile, scores: SignalScores, jd: JobDescription = None) -> str:
        """Generate a cohesive paragraph explaining the candidate's rank specifically."""
        reasons = []

        title = profile.job_titles[0] if profile.job_titles else "Professional"
        company = profile.companies[0] if profile.companies else "Unknown Company"
        exp = profile.experience_years
        
        # 1. Base profile statement
        base_stmt = f"{title} at {company} with {exp:.1f} years."
        reasons.append(base_stmt)

        # 2. Skill match specific statement
        if jd and jd.required_skills:
            c_canonical = {_canonicalize_skill(s) for s in profile.skills}
            req_canonical = {_canonicalize_skill(s) for s in jd.required_skills}
            matched_req = [s for s in jd.required_skills if _canonicalize_skill(s) in c_canonical]
            
            if len(matched_req) >= 3:
                reasons.append(f"Strong skill alignment including {', '.join(matched_req[:3])}.")
            elif len(matched_req) > 0:
                reasons.append(f"Partial skill match including {', '.join(matched_req)}.")

        # 3. Semantic / Domain
        sem = scores.semantic_similarity
        if sem >= 0.70:
            reasons.append("High semantic relevance to the ranking/retrieval domain.")
        elif sem >= 0.50:
            reasons.append("Moderate semantic match to the core JD.")

        if scores.career_evidence >= 0.7:
            reasons.append("Career history shows strong production retrieval, ranking, and evaluation evidence.")
        elif scores.career_evidence >= 0.4:
            reasons.append("Career history contains relevant applied search or ranking evidence.")

        if scores.skill_evidence >= 0.55:
            reasons.append("Relevant skills are supported by meaningful usage duration, proficiency, or assessments.")

        if scores.location_fit >= 0.8:
            reasons.append("Location is well aligned with the Pune/Noida hiring preference.")

        if scores.product_company_fit >= 0.7:
            reasons.append("Career history includes relevant product-company delivery experience.")
        elif scores.product_company_fit <= 0.2:
            reasons.append("Concern: career evidence is concentrated in services or consulting environments.")

        if scores.career_stability < 0.4:
            reasons.append("Concern: several short tenures reduce confidence in long-term fit.")

        # 4. Behavioral Signals (Redrob specific)
        behav = scores.behavioral_multiplier
        signals = profile.redrob_signals or {}
        
        behav_notes = []
        if behav > 1.2:
            behav_notes.append("highly active")
        
        notice = signals.get("notice_period_days", 60)
        if notice <= 30:
            behav_notes.append(f"{notice}-day notice")
            
        location = profile.location
        if location and jd and jd.location_preference:
            # simple check if any preferred location is in the candidate's location string
            is_pref = any(p.lower() in location.lower() for p in jd.location_preference.split(","))
            if is_pref:
                behav_notes.append(f"based in {location.split(',')[0]}")
                
        if behav_notes:
            reasons.append(f"Candidate is {', '.join(behav_notes)}.")

        if behav < 0.8:
            reasons.append("Warning: Low responsiveness or stale activity.")

        if scores.profile_trust < 0.7:
            reasons.append("Warning: Some skill claims have weak or inconsistent supporting evidence.")

        # Combine into a single string
        return " ".join(reasons)
