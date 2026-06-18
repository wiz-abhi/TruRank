"""
Signal computation — the core differentiator.

Computes signals for each candidate relative to a given JD,
combining semantic, behavioral, and India-specific heuristics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List

from src.jd_parser import JobDescription
from src.profile_parser import CandidateProfile
from src.utils import clamp, get_logger, load_config

logger = get_logger(__name__)


# ── India-specific skill synonym dictionary ──────────────────────────────
SKILL_SYNONYMS: Dict[str, List[str]] = {
    "machine learning": ["ml", "machine learning", "ml algorithms"],
    "deep learning": ["dl", "deep learning", "neural networks", "neural nets"],
    "natural language processing": [
        "nlp",
        "natural language processing",
        "text mining",
    ],
    "computer vision": ["cv", "computer vision", "image processing"],
    "artificial intelligence": ["ai", "artificial intelligence"],
    "python": ["python", "python3", "py"],
    "aws": ["aws", "amazon web services"],
    "gcp": ["gcp", "google cloud", "google cloud platform"],
    "azure": ["azure", "microsoft azure"],
    "docker": ["docker", "containerization", "containers"],
    "kubernetes": ["kubernetes", "k8s", "kube"],
    "elasticsearch": ["elasticsearch", "elastic", "es"],
    "vector database": ["vector db", "vectordb", "vector database"],
    "learning-to-rank": ["ltr", "learning to rank", "learning-to-rank"],
    "sentence-transformers": [
        "sentence transformers",
        "sentence-transformers",
        "sbert",
    ],
    "xgboost": ["xgboost", "xgb", "gradient boosting"],
}

# Build reverse lookup: synonym → canonical
_SYN_REVERSE: Dict[str, str] = {}
for canonical, syns in SKILL_SYNONYMS.items():
    for s in syns:
        _SYN_REVERSE[s.lower()] = canonical


def _canonicalize_skill(skill: str) -> str:
    """Map a skill string to its canonical form."""
    return _SYN_REVERSE.get(skill.lower().strip(), skill.lower().strip())


# ── Signal scores dataclass ──────────────────────────────────────────────
@dataclass
class SignalScores:
    """All computed signal scores for one candidate–JD pair."""

    semantic_similarity: float = 0.0
    skill_match: float = 0.0
    skill_recency: float = 0.0
    experience_fit: float = 0.0
    domain_alignment: float = 0.0
    culture_fit: float = 0.0
    education_tier_bonus: float = 0.0
    base_score: float = 0.0
    behavioral_multiplier: float = 1.0
    composite_score: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        """Return all scores as a dictionary."""
        return {
            "semantic_similarity": round(self.semantic_similarity, 4),
            "skill_match": round(self.skill_match, 4),
            "skill_recency": round(self.skill_recency, 4),
            "experience_fit": round(self.experience_fit, 4),
            "domain_alignment": round(self.domain_alignment, 4),
            "culture_fit": round(self.culture_fit, 4),
            "education_tier_bonus": round(self.education_tier_bonus, 4),
            "base_score": round(self.base_score, 4),
            "behavioral_multiplier": round(self.behavioral_multiplier, 4),
            "composite_score": round(self.composite_score, 4),
        }


# ── Signal computer ─────────────────────────────────────────────────────
class SignalComputer:
    """Compute ranking signals for candidate–JD pairs."""

    def __init__(self) -> None:
        self._cfg = load_config()
        self._weights = self._cfg.get("weights", {})
        self._edu_tiers = self._cfg.get("education_tiers", {})
        self._culture_cfg = self._cfg.get("culture", {})
        self._domain_cfg = self._cfg.get("domains", {})
        self._exp_cfg = self._cfg.get("experience", {})
        self._recency_cfg = self._cfg.get("skill_recency", {})
        self._behav_cfg = self._cfg.get("behavioral", {})

    # ── composite scorer ─────────────────────────────────────────────
    def compute_all(
        self,
        profile: CandidateProfile,
        jd: JobDescription,
        semantic_sim: float,
    ) -> SignalScores:
        """Compute all signals and the final composite score."""
        scores = SignalScores()

        scores.semantic_similarity = clamp(semantic_sim)
        scores.skill_match = self.skill_match_score(profile.skills, jd)
        scores.skill_recency = self.skill_recency_score(profile, jd)
        scores.experience_fit = self.experience_fit_score(
            profile.experience_years, jd.min_experience_years
        )
        scores.domain_alignment = self.domain_alignment_score(profile, jd)
        scores.culture_fit = self.culture_fit_score(profile, jd)
        scores.education_tier_bonus = self.education_tier_bonus(profile.education_tier)

        w = self._weights
        base = clamp(
            w.get("semantic_similarity", 0.35) * scores.semantic_similarity
            + w.get("skill_match", 0.25) * scores.skill_match
            + w.get("experience_fit", 0.15) * scores.experience_fit
            + w.get("domain_alignment", 0.10) * scores.domain_alignment
            + w.get("culture_fit", 0.10) * scores.culture_fit
            + w.get("skill_recency", 0.05) * scores.skill_recency
            + scores.education_tier_bonus,
            lo=0.0,
            hi=1.15,
        )
        scores.base_score = base

        # Behavioral multiplier
        behav_mult = self.behavioral_multiplier(profile)
        scores.behavioral_multiplier = behav_mult

        # Penalize if they ONLY have IT services experience and NO product experience
        services_only_penalty = self.services_only_penalty(profile)

        # Penalize if title is definitely not related to AI/ML/Data/Backend
        title_penalty = self.title_relevance_penalty(profile)

        final_score = base * behav_mult * services_only_penalty * title_penalty
        scores.composite_score = clamp(final_score, lo=0.0, hi=1.0)

        return scores

    # ── Behavioral Multiplier ────────────────────────────────────────
    def behavioral_multiplier(self, profile: CandidateProfile) -> float:
        """Compute a multiplier based on Redrob signals."""
        signals = profile.redrob_signals
        if not signals:
            return 1.0

        mult = self._behav_cfg.get("base_multiplier", 1.0)

        # 1. Recency
        last_active = profile.last_active
        if last_active:
            days_ago = (datetime.now() - last_active).days
            if days_ago <= 60:
                mult += self._behav_cfg.get("active_within_60d_bonus", 0.1)
            elif days_ago > 365:
                mult += self._behav_cfg.get("stale_penalty_365d", -0.6)
            elif days_ago > 180:
                mult += self._behav_cfg.get("stale_penalty_180d", -0.3)

        # 2. Response Rate
        response_rate = signals.get("recruiter_response_rate", 0.5)
        if response_rate < self._behav_cfg.get("low_response_penalty_threshold", 0.2):
            mult -= 0.3
        elif response_rate > self._behav_cfg.get("high_response_bonus_threshold", 0.7):
            mult += 0.1

        # 3. Not looking
        if not signals.get("open_to_work_flag", True):
            mult -= 0.2

        return clamp(mult, lo=0.1, hi=1.5)

    # ── Penalties ────────────────────────────────────────────────────
    def services_only_penalty(self, profile: CandidateProfile) -> float:
        """Penalize candidates with only IT Services background (per JD)."""
        enterprise_cos = {
            c.lower() for c in self._culture_cfg.get("enterprise_companies", [])
        }
        candidate_cos = {c.lower() for c in profile.companies}

        if not candidate_cos:
            return 1.0

        # If all companies are in the enterprise/services list
        if candidate_cos.issubset(enterprise_cos):
            return 0.7  # 30% penalty

        return 1.0

    def title_relevance_penalty(self, profile: CandidateProfile) -> float:
        """Penalize if the candidate has an explicitly unrelated title."""
        if not profile.job_titles:
            return 1.0

        current = profile.job_titles[0].lower()
        unrelated = [
            "marketing",
            "sales",
            "hr",
            "human resources",
            "accountant",
            "graphic",
            "civil",
            "mechanical",
        ]
        for u in unrelated:
            if u in current:
                return 0.3  # Heavy penalty for completely unrelated title (e.g. Marketing Manager)
        return 1.0

    # ── Original Signals ─────────────────────────────────────────────
    def skill_match_score(
        self, candidate_skills: List[str], jd: JobDescription
    ) -> float:
        if not jd.required_skills and not jd.preferred_skills:
            return 0.5

        c_canonical = {_canonicalize_skill(s) for s in candidate_skills}
        req_canonical = {_canonicalize_skill(s) for s in jd.required_skills}
        pref_canonical = {_canonicalize_skill(s) for s in jd.preferred_skills}

        req_matched = len(c_canonical & req_canonical)
        pref_matched = len(c_canonical & pref_canonical)

        total_req = max(len(req_canonical), 1)
        total_pref = max(len(pref_canonical), 1)

        score = (
            2.0 * (req_matched / total_req) + 1.0 * (pref_matched / total_pref)
        ) / 3.0
        return clamp(score)

    def skill_recency_score(
        self, profile: CandidateProfile, jd: JobDescription
    ) -> float:
        all_jd_skills = set(jd.required_skills + jd.preferred_skills)
        if not all_jd_skills:
            return 0.5

        threshold = self._recency_cfg.get("stale_threshold_years", 3)
        decay = self._recency_cfg.get("decay_factor", 0.15)
        current_year = datetime.now().year

        total_score = 0.0
        count = 0
        for skill in all_jd_skills:
            canon = _canonicalize_skill(skill)
            if canon in {_canonicalize_skill(s) for s in profile.skills}:
                count += 1
                last_year = profile.skill_recency.get(skill, current_year)
                years_ago = current_year - last_year
                if years_ago <= threshold:
                    total_score += 1.0
                else:
                    total_score += max(0.0, 1.0 - decay * (years_ago - threshold))

        if count == 0:
            return 0.0
        return clamp(total_score / max(len(all_jd_skills), 1))

    def experience_fit_score(self, candidate_exp: float, required_exp: float) -> float:
        if required_exp <= 0:
            return 0.8
        diff = candidate_exp - required_exp

        if abs(diff) < 0.5:
            return self._exp_cfg.get("exact_match_score", 1.0)
        elif abs(diff) <= 2.0:
            return self._exp_cfg.get("within_1_year_score", 0.9)
        elif diff > 2.0:
            penalty = min(0.3, diff * 0.03)
            return clamp(self._exp_cfg.get("overqualified_score", 0.8) - penalty)
        else:
            penalty_per = self._exp_cfg.get("underqualified_penalty_per_year", 0.15)
            return clamp(1.0 - abs(diff) * penalty_per)

    def domain_alignment_score(
        self, profile: CandidateProfile, jd: JobDescription
    ) -> float:
        if not jd.domain or jd.domain == "general":
            return 0.5
        domain_keywords = [kw.lower() for kw in self._domain_cfg.get(jd.domain, [])]
        if not domain_keywords:
            return 0.5

        text = " ".join(profile.job_titles + profile.companies + profile.skills).lower()
        matches = sum(1 for kw in domain_keywords if kw in text)
        return clamp(matches / max(len(domain_keywords) * 0.4, 1))

    def culture_fit_score(self, profile: CandidateProfile, jd: JobDescription) -> float:
        if not jd.culture_signals:
            return 0.5

        startup_signals = {"startup mindset", "fast-paced", "builder mentality"}
        jd_is_startup = bool(set(jd.culture_signals) & startup_signals)

        startup_cos = {
            c.lower() for c in self._culture_cfg.get("startup_companies", [])
        }
        candidate_cos = {c.lower() for c in profile.companies}
        startup_match = len(candidate_cos & startup_cos)

        if jd_is_startup:
            if startup_match > 0:
                return clamp(0.7 + startup_match * 0.1)
            return 0.5

        return 0.5

    def education_tier_bonus(self, tier: str) -> float:
        return self._edu_tiers.get(tier, {}).get("bonus", 0.0)
