"""
Job Description parser — extracts structured fields from free-text JDs
using regex/NLP heuristics with optional LLM enhancement.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.utils import get_logger, load_config, safe_float

logger = get_logger(__name__)


# ── Data model ────────────────────────────────────────────────────────────
@dataclass
class JobDescription:
    """Structured representation of a parsed job description."""

    raw_text: str = ""
    required_skills: List[str] = field(default_factory=list)
    preferred_skills: List[str] = field(default_factory=list)
    min_experience_years: float = 0.0
    seniority_level: str = "mid"
    domain: str = ""
    culture_signals: List[str] = field(default_factory=list)
    location_preference: str = ""
    role_summary: str = ""

    def to_embedding_text(self) -> str:
        """Build descriptive text for semantic embedding."""
        parts: List[str] = []
        if self.role_summary:
            parts.append(self.role_summary)
        if self.required_skills:
            parts.append(f"Required skills: {', '.join(self.required_skills)}.")
        if self.preferred_skills:
            parts.append(f"Nice-to-have skills: {', '.join(self.preferred_skills)}.")
        if self.domain:
            parts.append(f"Domain: {self.domain}.")
        if self.seniority_level:
            parts.append(f"Seniority: {self.seniority_level}.")
        if self.min_experience_years:
            parts.append(f"Minimum experience: {self.min_experience_years:.0f} years.")
        if self.location_preference:
            parts.append(f"Location: {self.location_preference}.")
        if self.culture_signals:
            parts.append(f"Culture: {', '.join(self.culture_signals)}.")
        return " ".join(parts) if parts else self.raw_text


# ── Parser ────────────────────────────────────────────────────────────────
class JDParser:
    """Parse raw JD text (or dict) into a ``JobDescription`` dataclass.

    Strategy:
        1. If an LLM client is available, use structured extraction.
        2. Otherwise fall back to regex + keyword heuristics.
    """

    # ── Seniority keywords ───────────────────────────────────────────
    _SENIORITY_MAP = {
        "intern": "intern",
        "entry": "junior",
        "junior": "junior",
        "associate": "junior",
        "mid": "mid",
        "senior": "senior",
        "sr": "senior",
        "staff": "senior",
        "principal": "lead",
        "lead": "lead",
        "manager": "lead",
        "director": "lead",
        "head": "lead",
        "vp": "lead",
        "cto": "lead",
    }

    # ── Culture signal keywords ──────────────────────────────────────
    _CULTURE_KEYWORDS: Dict[str, str] = {
        "move fast": "fast-paced",
        "fast-paced": "fast-paced",
        "startup": "startup mindset",
        "ownership": "ownership culture",
        "entrepreneurial": "startup mindset",
        "scrappy": "startup mindset",
        "hustle": "startup mindset",
        "zero to one": "builder mentality",
        "0 to 1": "builder mentality",
        "collaborative": "collaborative",
        "team player": "collaborative",
        "process-driven": "enterprise culture",
        "enterprise": "enterprise culture",
        "structured": "enterprise culture",
        "remote": "remote-friendly",
        "work from home": "remote-friendly",
        "hybrid": "hybrid work",
    }

    def __init__(self, llm_client: Any = None) -> None:
        """Initialise with optional LLM client for richer extraction.

        Args:
            llm_client: An Anthropic/OpenAI client instance (optional).
        """
        self._llm = llm_client
        self._cfg = load_config()
        self._domains = self._cfg.get("domains", {})

    # ── public API ───────────────────────────────────────────────────
    def parse(self, source: str | Dict[str, Any]) -> JobDescription:
        """Parse a JD from free text or a dictionary.

        Args:
            source: Either raw JD text or a dict with JD fields.

        Returns:
            Structured ``JobDescription``.
        """
        if isinstance(source, dict):
            return self._parse_dict(source)
        return self._parse_text(source)

    # ── dict-based parsing ───────────────────────────────────────────
    def _parse_dict(self, data: Dict[str, Any]) -> JobDescription:
        """Parse when the JD arrives as a structured dictionary."""
        raw_text = data.get("description", data.get("jd_text", json.dumps(data)))

        required = self._to_list(data.get("required_skills", data.get("skills", [])))
        preferred = self._to_list(
            data.get("preferred_skills", data.get("nice_to_have", []))
        )
        exp = safe_float(data.get("min_experience_years", data.get("experience", 0)))
        seniority = self._detect_seniority(
            str(data.get("seniority_level", data.get("title", "")))
        )
        domain = data.get("domain", self._detect_domain(raw_text))
        location = str(data.get("location", data.get("location_preference", "")))
        role_summary = str(data.get("role_summary", data.get("title", "")))

        culture = self._detect_culture(raw_text)

        return JobDescription(
            raw_text=raw_text,
            required_skills=required,
            preferred_skills=preferred,
            min_experience_years=exp,
            seniority_level=seniority,
            domain=domain,
            culture_signals=culture,
            location_preference=location,
            role_summary=role_summary,
        )

    # ── free-text parsing ────────────────────────────────────────────
    def _parse_text(self, text: str) -> JobDescription:
        """Extract structured fields from raw JD text using heuristics."""
        required = self._extract_skills_section(text, "required")
        preferred = self._extract_skills_section(text, "preferred")
        exp = self._extract_experience(text)
        seniority = self._detect_seniority(text)
        domain = self._detect_domain(text)
        culture = self._detect_culture(text)
        location = self._extract_location(text)
        role_summary = self._make_summary(text)

        jd = JobDescription(
            raw_text=text,
            required_skills=required,
            preferred_skills=preferred,
            min_experience_years=exp,
            seniority_level=seniority,
            domain=domain,
            culture_signals=culture,
            location_preference=location,
            role_summary=role_summary,
        )
        logger.info(
            "Parsed JD — domain=%s, seniority=%s, required_skills=%d, exp=%.1f yrs",
            jd.domain,
            jd.seniority_level,
            len(jd.required_skills),
            jd.min_experience_years,
        )
        return jd

    # ── heuristic helpers ────────────────────────────────────────────
    @staticmethod
    def _extract_skills_section(text: str, kind: str) -> List[str]:
        """Pull skill lists from bullet-pointed or comma-separated sections."""
        patterns = {
            "required": [
                r"(?:required|must.have|mandatory|key)\s*(?:skills?|qualifications?|requirements?)\s*[:\-]?\s*(.*?)(?:\n\n|\Z)",
                r"(?:requirements?|what.you.need)\s*[:\-]?\s*(.*?)(?:\n\n|\Z)",
            ],
            "preferred": [
                r"(?:preferred|nice.to.have|good.to.have|bonus|desired)\s*(?:skills?|qualifications?)?\s*[:\-]?\s*(.*?)(?:\n\n|\Z)",
            ],
        }
        for pat in patterns.get(kind, []):
            m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
            if m:
                block = m.group(1)
                items = re.split(r"[\n•\-\*,;]+", block)
                skills = [s.strip().strip("- •*") for s in items if len(s.strip()) > 1]
                return skills[:20]

        # Fallback: scan for common tech keywords
        if kind == "required":
            return JDParser._keyword_scan(text)
        return []

    @staticmethod
    def _keyword_scan(text: str) -> List[str]:
        """Scan for common tech skills in the text."""
        _TECH_KEYWORDS = [
            "Python",
            "Java",
            "JavaScript",
            "TypeScript",
            "Go",
            "Golang",
            "Rust",
            "C++",
            "C#",
            "Ruby",
            "PHP",
            "Scala",
            "Kotlin",
            "Swift",
            "React",
            "Angular",
            "Vue",
            "Node.js",
            "Django",
            "Flask",
            "FastAPI",
            "Spring",
            "Spring Boot",
            ".NET",
            "AWS",
            "GCP",
            "Azure",
            "Docker",
            "Kubernetes",
            "Terraform",
            "PostgreSQL",
            "MySQL",
            "MongoDB",
            "Redis",
            "Elasticsearch",
            "Kafka",
            "RabbitMQ",
            "Spark",
            "Hadoop",
            "Airflow",
            "TensorFlow",
            "PyTorch",
            "scikit-learn",
            "Pandas",
            "NumPy",
            "Machine Learning",
            "Deep Learning",
            "NLP",
            "Computer Vision",
            "MLOps",
            "CI/CD",
            "Git",
            "Jenkins",
            "GitHub Actions",
            "REST",
            "GraphQL",
            "gRPC",
            "Microservices",
            "SQL",
            "NoSQL",
            "ETL",
            "Data Pipeline",
            "Figma",
            "Sketch",
            "Adobe XD",
            "Agile",
            "Scrum",
            "Jira",
        ]
        found: List[str] = []
        text_lower = text.lower()
        for kw in _TECH_KEYWORDS:
            if kw.lower() in text_lower:
                found.append(kw)
        return found[:15]

    @staticmethod
    def _extract_experience(text: str) -> float:
        """Extract minimum years of experience from JD text."""
        patterns = [
            r"(\d+\.?\d*)\s*\+?\s*(?:years?|yrs?)\s*(?:of)?\s*(?:experience|exp)",
            r"(?:minimum|min|at\s*least)\s*(\d+\.?\d*)\s*(?:years?|yrs?)",
            r"(\d+)\s*-\s*\d+\s*(?:years?|yrs?)",
        ]
        for pat in patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                return float(m.group(1))
        return 0.0

    def _detect_seniority(self, text: str) -> str:
        """Detect seniority level from text."""
        text_lower = text.lower()
        for keyword, level in self._SENIORITY_MAP.items():
            if keyword in text_lower:
                return level
        return "mid"

    def _detect_domain(self, text: str) -> str:
        """Detect primary domain by keyword frequency."""
        text_lower = text.lower()
        scores: Dict[str, int] = {}
        for domain, keywords in self._domains.items():
            score = sum(1 for kw in keywords if kw.lower() in text_lower)
            if score > 0:
                scores[domain] = score
        if scores:
            return max(scores, key=scores.get)  # type: ignore[arg-type]
        return "general"

    def _detect_culture(self, text: str) -> List[str]:
        """Detect culture signals from JD text."""
        text_lower = text.lower()
        signals: List[str] = []
        seen: set[str] = set()
        for keyword, signal in self._CULTURE_KEYWORDS.items():
            if keyword in text_lower and signal not in seen:
                signals.append(signal)
                seen.add(signal)
        return signals

    @staticmethod
    def _extract_location(text: str) -> str:
        """Try to extract location from JD text."""
        _INDIAN_CITIES = [
            "Bangalore",
            "Bengaluru",
            "Mumbai",
            "Delhi",
            "NCR",
            "Gurgaon",
            "Gurugram",
            "Noida",
            "Hyderabad",
            "Chennai",
            "Pune",
            "Kolkata",
            "Ahmedabad",
            "Jaipur",
            "Kochi",
            "Coimbatore",
            "Indore",
            "Chandigarh",
            "Lucknow",
            "Thiruvananthapuram",
            "Remote",
        ]
        for city in _INDIAN_CITIES:
            if city.lower() in text.lower():
                return city
        return ""

    @staticmethod
    def _make_summary(text: str) -> str:
        """Create a one-sentence summary from the first meaningful line."""
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        for line in lines[:5]:
            if len(line) > 20:
                return line[:200]
        return text[:200] if text else ""

    @staticmethod
    def _to_list(value: Any) -> List[str]:
        """Convert various inputs to a list of strings."""
        if isinstance(value, list):
            return [str(v).strip() for v in value if str(v).strip()]
        if isinstance(value, str):
            return [s.strip() for s in re.split(r"[,;|]+", value) if s.strip()]
        return []
