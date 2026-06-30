"""
Signal computation — the core differentiator.

Computes signals for each candidate relative to a given JD,
combining semantic, behavioral, and India-specific heuristics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List

REFERENCE_DATE = date(2026, 6, 22)

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
    career_stability: float = 0.0
    product_company_fit: float = 0.0
    work_mode_fit: float = 0.0
    education_tier_bonus: float = 0.0
    base_score: float = 0.0
    behavioral_multiplier: float = 1.0
    composite_score: float = 0.0
    cross_encoder_score: float = -1.0   # -1 means CE was not run
    skill_corroboration: float = 0.0    # fraction of claimed skills corroborated by career text
    external_validation: float = 0.0
    production_recency: float = 0.0

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
            "career_stability": round(self.career_stability, 4),
            "product_company_fit": round(self.product_company_fit, 4),
            "work_mode_fit": round(self.work_mode_fit, 4),
            "education_tier_bonus": round(self.education_tier_bonus, 4),
            "external_validation": round(self.external_validation, 4),
            "production_recency": round(self.production_recency, 4),
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
        scores.career_stability = self.career_stability_score(profile)
        scores.product_company_fit = self.product_company_fit_score(profile)
        scores.work_mode_fit = self.work_mode_fit_score(profile)
        scores.education_tier_bonus = self.education_tier_bonus(profile.education_tier)
        scores.skill_corroboration = self.skill_corroboration_score(profile, jd)
        scores.external_validation = self.external_validation_score(profile)
        scores.production_recency = self.production_recency_score(profile)

        w = self._weights
        # Weights optimized by silver-label sweep (tools/weight_sweep.py)
        # Baseline NDCG@10: 0.879 → Optimized: 0.952 (+8.3%)
        base = (
            w.get("semantic_similarity", 0.0951) * scores.semantic_similarity
            + w.get("skill_match", 0.0140) * scores.skill_match
            + w.get("skill_evidence", 0.0815) * scores.skill_evidence
            + w.get("skill_corroboration", 0.0318) * scores.skill_corroboration
            + w.get("career_evidence", 0.1998) * scores.career_evidence
            + w.get("experience_fit", 0.0751) * scores.experience_fit
            + w.get("domain_alignment", 0.0702) * scores.domain_alignment
            + w.get("culture_fit", 0.0187) * scores.culture_fit
            + w.get("location_fit", 0.1000) * scores.location_fit
            + w.get("skill_recency", 0.0458) * scores.skill_recency
            + w.get("career_stability", 0.0388) * scores.career_stability
            + w.get("product_company_fit", 0.0900) * scores.product_company_fit
            + w.get("work_mode_fit", 0.0501) * scores.work_mode_fit
            + w.get("external_validation", 0.0287) * scores.external_validation
            + w.get("production_recency", 0.0604) * scores.production_recency
        )
        scores.base_score = clamp(base, 0.0, 1.0)

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
        scores.composite_score = clamp(final_score, 0.0, 1.0)

        return scores

    # ── Behavioral Multiplier (23 Redrob signals, bounded 0.50–1.15×) ────────
    def behavioral_multiplier(self, profile: CandidateProfile) -> float:
        """Compute a bounded multiplier based on 23 Redrob signals.

        Range is deliberately asymmetric: floor 0.50 (heavily penalize unavailable),
        ceiling 1.15 (mildly reward highly active). This matches Caliber's design:
        availability can push a strong candidate down or up, but never manufacture
        relevance or erase it entirely.
        """
        signals = profile.redrob_signals
        if not signals:
            return 1.0

        mult = self._behav_cfg.get("base_multiplier", 1.0)

        # 1. Recency / last active
        last_active = profile.last_active
        if last_active:
            days_ago = (REFERENCE_DATE - last_active.date()).days
            if days_ago <= 30:
                mult += 0.12   # very recently active
            elif days_ago <= 60:
                mult += 0.08
            elif days_ago > 365:
                mult -= 0.6    # stale profile
            elif days_ago > 180:
                mult -= 0.3

        # 2. Recruiter response rate
        response_rate = signals.get("recruiter_response_rate", 0.5)
        if response_rate < 0.15:
            mult -= 0.35
        elif response_rate < 0.3:
            mult -= 0.15
        elif response_rate > 0.8:
            mult += 0.1
        elif response_rate > 0.6:
            mult += 0.05

        # 3. Open to work
        if not signals.get("open_to_work_flag", True):
            mult -= 0.25

        # 4. Notice period (JD prefers sub-30 days)
        notice_period = signals.get("notice_period_days", 60)
        if notice_period <= 15:
            mult += 0.1
        elif notice_period <= 30:
            mult += 0.06
        elif notice_period > 90:
            mult -= 0.2
        elif notice_period > 60:
            mult -= 0.08

        # 5. GitHub activity score (−1 means no GitHub linked: treat as neutral)
        github_score = signals.get("github_activity_score", -1)
        if github_score >= 70:
            mult += 0.15
        elif github_score >= 40:
            mult += 0.08
        # github_score == -1 (~65% of pool): neutral, no penalty

        # 6. Profile completeness
        completeness = signals.get("profile_completeness_score", 100)
        if completeness >= 90:
            mult += 0.03
        elif completeness < 30:
            mult -= 0.25
        elif completeness < 50:
            mult -= 0.12

        # 7. Average response time
        avg_resp_hours = signals.get("avg_response_time_hours", 48)
        if avg_resp_hours < 12:
            mult += 0.06
        elif avg_resp_hours < 24:
            mult += 0.03
        elif avg_resp_hours > 168:  # > 1 week
            mult -= 0.08

        # 8. Interview completion rate
        int_completion = signals.get("interview_completion_rate", 1.0)
        if int_completion >= 0.9:
            mult += 0.03
        elif int_completion < 0.3:
            mult -= 0.2
        elif int_completion < 0.5:
            mult -= 0.1

        # 9. Saved by recruiters (demand signal)
        saved = signals.get("saved_by_recruiters_30d", 0)
        if saved >= 10:
            mult += 0.08
        elif saved >= 5:
            mult += 0.04

        # 10 & 11. Verifications
        verified_email = signals.get("verified_email", False)
        verified_phone = signals.get("verified_phone", False)
        if verified_email and verified_phone:
            mult += 0.03
        elif not verified_email and not verified_phone:
            mult -= 0.12

        # 12. LinkedIn connected
        if signals.get("linkedin_connected", False):
            mult += 0.04

        # 13. Willing to relocate (relevant if not in preferred location)
        if signals.get("willing_to_relocate", False):
            mult += 0.03

        # 14. Preferred work mode alignment
        work_mode = str(signals.get("preferred_work_mode", "")).lower()
        if work_mode in ("hybrid", "flexible"):
            mult += 0.03

        # 15. Applications sent (engagement signal)
        apps = signals.get("applications_sent_30d", 0)
        if apps >= 10:
            mult += 0.04
        elif apps == 0:
            mult -= 0.04

        # 16. Profile views (market interest)
        views = signals.get("profile_views_30d", 0)
        if views >= 20:
            mult += 0.04

        # 17. Certifications count
        certs = signals.get("certifications_count", 0)
        if certs >= 3:
            mult += 0.03

        # 18. Endorsements received
        endorsements = signals.get("endorsements_received_total", 0)
        if endorsements >= 30:
            mult += 0.03

        # 19. Referral available
        if signals.get("referral_available", False):
            mult += 0.03

        # 20. Expected salary alignment (if provided)
        salary_match = signals.get("salary_expectation_match", None)
        if salary_match is not None:
            if salary_match < 0.6:
                mult -= 0.05

        # 21. Last job change recency
        months_in_current = signals.get("months_in_current_role", 0)
        if 6 <= months_in_current <= 12:
            mult -= 0.04  # just joined, unlikely to move

        # 22. Portfolio/projects linked
        if signals.get("portfolio_linked", False):
            mult += 0.03

        # 23. Screening completion
        screening = signals.get("screening_completion_rate", 1.0)
        if screening < 0.3:
            mult -= 0.08

        # Bounded: availability should re-rank but never manufacture or erase relevance.
        return clamp(mult, lo=0.50, hi=1.15)

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

    def skill_corroboration_score(self, profile: CandidateProfile, jd: JobDescription) -> float:
        """Caliber-style skill gate: skills only earn credit if career descriptions corroborate them.

        A candidate listing 'pinecone' as a skill earns no credit if none of their role
        descriptions mention vector databases. This naturally sinks keyword stuffers.
        """
        required = {_canonicalize_skill(s) for s in jd.required_skills}
        claimed_required = [
            s for s in profile.skills
            if _canonicalize_skill(s) in required
        ]
        if not claimed_required:
            return 0.0

        # Career description corpus for this candidate
        desc_text = " ".join(profile.career_history_desc).lower()

        # Synonyms for key terms we look for in descriptions
        CORROBORATION_TERMS = {
            "retrieval": ["retrieval", "search", "recall", "fetch", "index"],
            "ranking": ["ranking", "rank", "rerank", "ltr", "learning to rank"],
            "embedding": ["embedding", "embed", "vector", "encode", "encode"],
            "semantic": ["semantic", "meaning", "intent", "similarity"],
            "recommendation": ["recommendation", "recommend", "suggest", "collaborative"],
            "faiss": ["faiss", "vector index", "ann", "approximate nearest"],
            "pinecone": ["pinecone", "vector db", "vector database"],
            "qdrant": ["qdrant", "vector store", "vector search"],
            "weaviate": ["weaviate", "vector search"],
            "milvus": ["milvus", "vector database"],
            "elasticsearch": ["elasticsearch", "elastic", "search engine"],
            "sentence-transformers": ["sentence-transformers", "sbert", "bi-encoder"],
            "python": ["python"],
        }

        corroborated = 0
        for skill in claimed_required:
            canonical = _canonicalize_skill(skill)
            terms = CORROBORATION_TERMS.get(canonical, [canonical.lower()])
            if any(t in desc_text for t in terms):
                corroborated += 1

        return corroborated / len(claimed_required)

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

    def work_mode_fit_score(self, profile: CandidateProfile) -> float:
        mode = str(profile.redrob_signals.get("preferred_work_mode", "")).lower()
        if mode in ("hybrid", "flexible"):
            return 1.0
        if mode == "onsite":
            return 0.8 if self.location_fit_score(profile) >= 0.8 else 0.45
        if mode == "remote":
            return 0.55
        return 0.5

    def career_stability_score(self, profile: CandidateProfile) -> float:
        """Reward sustained delivery and penalize repeated short title-chasing moves."""
        roles = [r for r in profile.raw.get("career_history", []) if isinstance(r, dict)]
        if not roles:
            return 0.4
        durations = [float(r.get("duration_months", 0) or 0) for r in roles]
        short = sum(0 < months < 20 for months in durations)
        sustained = sum(months >= 30 for months in durations)
        senior_words = ("senior", "staff", "principal", "lead")
        title_chasing = 0
        for previous, current in zip(roles[1:], roles[:-1]):
            if float(current.get("duration_months", 0) or 0) < 20:
                old_level = sum(word in str(previous.get("title", "")).lower() for word in senior_words)
                new_level = sum(word in str(current.get("title", "")).lower() for word in senior_words)
                title_chasing += new_level > old_level
        score = 0.55 + min(0.3, sustained * 0.12) - min(0.45, short * 0.1 + title_chasing * 0.12)
        return clamp(score)

    def product_company_fit_score(self, profile: CandidateProfile) -> float:
        """Use supplied industry and company-size metadata, not only name lists."""
        roles = [r for r in profile.raw.get("career_history", []) if isinstance(r, dict)]
        if not roles:
            return 0.4
        services_terms = ("it services", "consulting", "outsourcing", "professional services")
        product_terms = ("software product", "internet", "e-commerce", "fintech", "hr tech", "marketplace", "saas")
        product = services = 0
        for role in roles:
            industry = str(role.get("industry", "")).lower()
            description = str(role.get("description", "")).lower()
            product += any(term in industry for term in product_terms)
            services += any(term in industry for term in services_terms)
            product += any(term in description for term in ("product team", "customers", "end users", "user engagement"))
        if product:
            return clamp(0.65 + 0.12 * product)
        if services == len(roles):
            return 0.1
        return 0.45

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
        roles = [r for r in profile.raw.get("career_history", []) if isinstance(r, dict)]
        services_industries = all(
            any(term in str(r.get("industry", "")).lower() for term in ("it services", "consulting", "outsourcing"))
            for r in roles
        ) if roles else False
        normalized_services = all(
            any(alias in company for alias in enterprise_cos)
            for company in candidate_cos
        )
        if normalized_services or services_industries:
            return 0.1  # 90% penalty (JD says this is a disqualifier)

        return 1.0

    def title_relevance_penalty(self, profile: CandidateProfile) -> float:
        """Penalize if the candidate has an explicitly unrelated title, using comprehensive regexes."""
        if not profile.job_titles:
            return 1.0

        current = profile.job_titles[0].lower()
        
        import re
        NONTECH_TITLE_RE = re.compile(
            r"\b(hr\b|human resource|recruit|talent acquisition|sales\b|marketing|content|"
            r"writer|graphic|design|account|finance|financial|mechanical|civil\b|"
            r"operations|teacher|nurse|doctor|lawyer|business analyst|customer\b|"
            r"support\b|administrat|product manager|project manager|consultant|executive\b)",
            re.I,
        )
        ADJACENT_TITLE_RE = re.compile(
            r"\b(data engineer|senior data engineer|analytics engineer|backend engineer|"
            r"software engineer|full[- ]?stack|sde\b|platform engineer|developer|programmer)\b",
            re.I,
        )
        STRONG_TITLE_RE = re.compile(
            r"\b(ml engineer|machine learning engineer|ai engineer|applied (ml |ai )?scientist|"
            r"ai research(er| engineer)?|research engineer|data scientist|ml scientist|"
            r"nlp engineer|ml ?ops engineer|deep learning|staff ml|senior ml|senior ai|"
            r"ai specialist|search engineer|relevance engineer|recommendation engineer)\b",
            re.I,
        )
        
        if NONTECH_TITLE_RE.search(current):
            return 0.1  # Heavy penalty for completely unrelated title
        elif STRONG_TITLE_RE.search(current):
            return 1.1  # Slight boost for perfectly aligned title
        elif ADJACENT_TITLE_RE.search(current):
            return 1.0
            
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

    def external_validation_score(self, profile: CandidateProfile) -> float:
        """
        Score based on GitHub activity, OSS contribution, or StackOverflow rep.
        Mirrors Caliber's external_validation feature.
        """
        score = 0.0
        github_score = profile.redrob_signals.get("github_activity_score", 0)
        if github_score > 70:
            score += 1.0
        elif github_score > 40:
            score += 0.5
        
        if profile.redrob_signals.get("open_source_contributor", False):
            score += 0.5
            
        if profile.redrob_signals.get("stackoverflow_reputation", 0) > 1000:
            score += 0.5
            
        return clamp(score, 0.0, 1.0)

    def production_recency_score(self, profile: CandidateProfile) -> float:
        """
        Score based on evidence of shipping/production in recent roles.
        Mirrors Caliber's production_recency feature.
        """
        score = 0.0
        prod_terms = ["production", "deployed", "shipped", "live system", "serving", "mlops", "inference"]
        
        # Check recent roles (first 2)
        recent_desc = " ".join(profile.career_history_desc[:2]).lower()
        if any(term in recent_desc for term in prod_terms):
            score += 1.0
        else:
            # Check older roles
            older_desc = " ".join(profile.career_history_desc[2:]).lower()
            if any(term in older_desc for term in prod_terms):
                score += 0.5
                
        return score
