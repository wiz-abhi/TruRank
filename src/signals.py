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
    career_evidence: float = 0.0
    skill_evidence: float = 0.0
    location_fit: float = 0.0
    profile_trust: float = 1.0
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
            "career_evidence": round(self.career_evidence, 4),
            "skill_evidence": round(self.skill_evidence, 4),
            "location_fit": round(self.location_fit, 4),
            "profile_trust": round(self.profile_trust, 4),
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
        scores.career_evidence = self.career_evidence_score(profile)
        scores.skill_evidence = self.skill_evidence_score(profile, jd)
        scores.location_fit = self.location_fit_score(profile)
        scores.profile_trust = self.profile_trust_score(profile)
        scores.education_tier_bonus = self.education_tier_bonus(profile.education_tier)

        w = self._weights
        base = (
            w.get("semantic_similarity", 0.18) * scores.semantic_similarity
            + w.get("skill_match", 0.08) * scores.skill_match
            + w.get("skill_evidence", 0.12) * scores.skill_evidence
            + w.get("career_evidence", 0.28) * scores.career_evidence
            + w.get("experience_fit", 0.10) * scores.experience_fit
            + w.get("domain_alignment", 0.08) * scores.domain_alignment
            + w.get("culture_fit", 0.06) * scores.culture_fit
            + w.get("location_fit", 0.05) * scores.location_fit
            + w.get("skill_recency", 0.05) * scores.skill_recency
        )
        scores.base_score = base

        # Behavioral multiplier
        behav_mult = self.behavioral_multiplier(profile)
        scores.behavioral_multiplier = behav_mult

        # Penalize if they ONLY have IT services experience and NO product experience
        services_only_penalty = self.services_only_penalty(profile)

        # Penalize if title is definitely not related to AI/ML/Data/Backend
        title_penalty = self.title_relevance_penalty(profile)

        final_score = (
            base * behav_mult * services_only_penalty * title_penalty
            * scores.profile_trust
        )
        scores.composite_score = max(0.0, final_score)

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
            
        # 4. Notice Period (JD prefers sub-30 days)
        notice_period = signals.get("notice_period_days", 60)
        if notice_period <= 30:
            mult += 0.1
        elif notice_period > 90:
            mult -= 0.15
            
        # 5. GitHub Activity Score
        github_score = signals.get("github_activity_score", -1)
        if github_score > 40:
            mult += 0.15
            
        # 6. Profile Completeness
        completeness = signals.get("profile_completeness_score", 100)
        if completeness < 40:
            mult -= 0.2
            
        # 7. Response Time
        avg_resp_hours = signals.get("avg_response_time_hours", 48)
        if avg_resp_hours < 24:
            mult += 0.05
            
        # 8. Interview Completion Rate
        int_completion = signals.get("interview_completion_rate", 1.0)
        if int_completion < 0.3:
            mult -= 0.2
            
        # 9. Saved by Recruiters
        saved = signals.get("saved_by_recruiters_30d", 0)
        if saved > 5:
            mult += 0.05
            
        # 10 & 11. Verifications
        verified_email = signals.get("verified_email", False)
        verified_phone = signals.get("verified_phone", False)
        if not verified_email and not verified_phone:
            mult -= 0.1
            
        # 12. LinkedIn Connected
        if signals.get("linkedin_connected", False):
            mult += 0.05

        # Availability can re-rank technically credible candidates, but should
        # never manufacture relevance or erase it entirely.
        return clamp(mult, lo=0.65, hi=1.25)

    def career_evidence_score(self, profile: CandidateProfile) -> float:
        """Score JD meaning from role history rather than skill keywords."""
        raw = profile.raw
        roles = raw.get("career_history", [])
        text = " ".join(
            " ".join(str(r.get(k, "")) for k in ("title", "industry", "description"))
            for r in roles if isinstance(r, dict)
        ).lower()
        current = (raw.get("profile", {}).get("current_title", "") or "").lower()

        retrieval = ["retrieval", "ranking", "search", "recommendation", "recommender", "bm25", "vector", "semantic search"]
        production = ["production", "deployed", "shipped", "real-time", "scale", "latency", "users", "pipeline", "monitoring"]
        evaluation = ["ndcg", "mrr", "map", "a/b", "offline evaluation", "benchmark", "experiment"]
        ownership = ["owned", "led", "designed", "architected", "mentored", "cross-functional", "product"]

        def coverage(words: List[str], cap: int) -> float:
            return min(1.0, sum(word in text for word in words) / cap)

        score = (
            0.38 * coverage(retrieval, 3)
            + 0.27 * coverage(production, 3)
            + 0.20 * coverage(evaluation, 2)
            + 0.15 * coverage(ownership, 2)
        )
        relevant_title = any(x in current for x in ("machine learning", "ml", "ai", "nlp", "search", "recommend", "data scientist", "applied scientist"))
        if relevant_title:
            score += 0.1
        research_only = roles and all(
            any(x in str(r.get("title", "")).lower() for x in ("researcher", "research scientist", "phd"))
            for r in roles if isinstance(r, dict)
        ) and not any(x in text for x in production)
        if research_only:
            score *= 0.25
        primary_mismatch = any(x in current for x in ("computer vision", "speech", "robotics")) and not coverage(retrieval, 2)
        if primary_mismatch:
            score *= 0.45
        return clamp(score)

    def skill_evidence_score(self, profile: CandidateProfile, jd: JobDescription) -> float:
        """Validate claimed skills using duration, proficiency and assessments."""
        required = {_canonicalize_skill(s) for s in jd.required_skills}
        assessments = {
            _canonicalize_skill(k): float(v)
            for k, v in profile.redrob_signals.get("skill_assessment_scores", {}).items()
        }
        evidence = []
        proficiency = {"beginner": 0.2, "intermediate": 0.5, "advanced": 0.8, "expert": 1.0}
        for skill in profile.raw.get("skills", []):
            if not isinstance(skill, dict):
                continue
            name = _canonicalize_skill(str(skill.get("name", "")))
            if name not in required:
                continue
            duration = min(1.0, float(skill.get("duration_months", 0)) / 36.0)
            endorsed = min(1.0, float(skill.get("endorsements", 0)) / 30.0)
            assessed = assessments.get(name, 50.0) / 100.0
            prof = proficiency.get(str(skill.get("proficiency", "")).lower(), 0.0)
            evidence.append(0.4 * duration + 0.25 * prof + 0.15 * endorsed + 0.2 * assessed)
        return sum(sorted(evidence, reverse=True)[:6]) / 6.0

    def location_fit_score(self, profile: CandidateProfile) -> float:
        location = profile.location.lower()
        signals = profile.redrob_signals
        if any(city in location for city in ("pune", "noida")):
            return 1.0
        if any(city in location for city in ("hyderabad", "mumbai", "delhi", "ncr", "gurgaon", "gurugram")):
            return 0.8
        if signals.get("willing_to_relocate"):
            return 0.65
        return 0.2 if profile.raw.get("profile", {}).get("country") == "India" else 0.0

    def profile_trust_score(self, profile: CandidateProfile) -> float:
        skills = [s for s in profile.raw.get("skills", []) if isinstance(s, dict)]
        if not skills:
            return 0.8
        zero_experts = sum(s.get("proficiency") == "expert" and s.get("duration_months", 0) == 0 for s in skills)
        trust = 1.0 - min(0.65, zero_experts * 0.12)
        return clamp(trust, 0.25, 1.0)

    # ── Penalties ────────────────────────────────────────────────────
    def services_only_penalty(self, profile: CandidateProfile) -> float:
        """Penalize candidates with only IT Services background (per JD)."""
        enterprise_cos = {
            c.lower() for c in self._culture_cfg.get("enterprise_companies", [])
        }
        candidate_cos = {c.lower() for c in profile.companies}

        if not candidate_cos:
            return 1.0

        # If all companies are in the enterprise/services list (JD Disqualifier)
        if candidate_cos.issubset(enterprise_cos):
            return 0.1  # 90% penalty (JD says this is a disqualifier)

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

        duration_by_skill = {
            _canonicalize_skill(name): months
            for name, months in profile.skill_recency.items()
        }
        total_score = 0.0
        count = 0
        for skill in all_jd_skills:
            canon = _canonicalize_skill(skill)
            if canon in {_canonicalize_skill(s) for s in profile.skills}:
                count += 1
                duration = duration_by_skill.get(canon, 0)
                total_score += min(1.0, duration / 36.0)

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
