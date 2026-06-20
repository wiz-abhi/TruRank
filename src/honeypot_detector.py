"""
Honeypot detector — identifies ~80 'subtly impossible' candidate profiles.

From the challenge docs:
  - "8 years of experience at a company founded 3 years ago"
  - "'expert' proficiency in 10 skills with 0 years used"
  - Submissions with honeypot rate > 10% in top 100 are disqualified.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class HoneypotResult:
    """Result of honeypot analysis for a single candidate."""
    candidate_id: str
    is_honeypot: bool
    severity: int  # total severity score
    flags: List[str] = field(default_factory=list)


def _parse_date(d: Any) -> Optional[date]:
    """Parse a date string, return None on failure."""
    if not d:
        return None
    try:
        return datetime.strptime(str(d).strip(), "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


class HoneypotDetector:
    """Detect honeypot candidates with impossible profile contradictions."""

    # Severity threshold: candidates at or above this are classified as honeypots
    HONEYPOT_THRESHOLD = 3

    def detect(self, raw: Dict[str, Any]) -> HoneypotResult:
        """Analyze a single raw candidate dict for honeypot signals.

        Args:
            raw: The full candidate JSON dict (with profile, career_history, etc.)

        Returns:
            HoneypotResult with is_honeypot flag and detailed flags.
        """
        cid = raw.get("candidate_id", "UNKNOWN")
        flags: List[Tuple[int, str]] = []  # (severity, description)

        profile = raw.get("profile", {})
        skills = raw.get("skills", [])
        career = raw.get("career_history", [])
        education = raw.get("education", [])
        signals = raw.get("redrob_signals", {})
        exp_years = profile.get("years_of_experience", 0) if isinstance(profile, dict) else 0
        today = date.today()

        # ── HARD FLAG 1: Expert proficiency with 0 months duration ──
        if isinstance(skills, list):
            expert_zero = [
                s for s in skills
                if isinstance(s, dict)
                and s.get("proficiency") == "expert"
                and s.get("duration_months", -1) == 0
            ]
            if len(expert_zero) >= 2:
                flags.append((3, f"{len(expert_zero)} skills listed as 'expert' with 0 months usage"))
            elif len(expert_zero) == 1:
                flags.append((1, f"1 skill listed as 'expert' with 0 months usage: {expert_zero[0].get('name')}"))

        # ── HARD FLAG 2: Role duration vastly exceeds calendar span ──
        if isinstance(career, list):
            for role in career:
                if not isinstance(role, dict):
                    continue
                start = _parse_date(role.get("start_date"))
                end = _parse_date(role.get("end_date"))
                dur_months = role.get("duration_months", 0)
                if start and end and dur_months > 0:
                    actual_months = (end.year - start.year) * 12 + (end.month - start.month)
                    if actual_months > 0:
                        ratio = dur_months / actual_months
                        if ratio > 2.0 and dur_months > 24:
                            flags.append((3, f"Role at {role.get('company', '?')}: claims {dur_months}mo but dates span only {actual_months}mo"))
                        elif ratio > 1.8 and dur_months > 12:
                            flags.append((2, f"Role at {role.get('company', '?')}: claims {dur_months}mo but dates span only {actual_months}mo (ratio {ratio:.1f}x)"))

        # ── HARD FLAG 3: Career duration exceeds time since start date ──
        if isinstance(career, list):
            for role in career:
                if not isinstance(role, dict):
                    continue
                start = _parse_date(role.get("start_date"))
                dur_months = role.get("duration_months", 0)
                if start and dur_months >= 12:
                    today = date.today()
                    months_available = (today.year - start.year) * 12 + (today.month - start.month)
                    if dur_months > months_available + 3:
                        severity = 3 if dur_months > months_available + 12 else 2
                        flags.append((severity, f"Role at {role.get('company', '?')}: claims {dur_months}mo but only {months_available}mo since start"))

        # ── HARD FLAG 4: End date before start date ──
        if isinstance(career, list):
            for role in career:
                if not isinstance(role, dict):
                    continue
                start = _parse_date(role.get("start_date"))
                end = _parse_date(role.get("end_date"))
                if start and end and end < start:
                    flags.append((3, f"Role at {role.get('company', '?')}: end_date {end} before start_date {start}"))
                if start and start > today:
                    flags.append((3, f"Role at {role.get('company', '?')}: future start date {start}"))
                if end and end > today:
                    flags.append((2, f"Role at {role.get('company', '?')}: future end date {end}"))

        # Overlapping full-time roles are possible, but multiple long overlaps
        # combined with inflated durations are a strong consistency warning.
        dated_roles = []
        if isinstance(career, list):
            for role in career:
                if not isinstance(role, dict):
                    continue
                start = _parse_date(role.get("start_date"))
                end = _parse_date(role.get("end_date")) or today
                if start and end:
                    dated_roles.append((start, end, role.get("company", "?")))
        long_overlaps = 0
        for idx, (start_a, end_a, _) in enumerate(dated_roles):
            for start_b, end_b, _ in dated_roles[idx + 1:]:
                overlap_days = (min(end_a, end_b) - max(start_a, start_b)).days
                long_overlaps += overlap_days > 365
        if long_overlaps >= 2:
            flags.append((3, f"{long_overlaps} career-role overlaps longer than one year"))
        elif long_overlaps == 1:
            flags.append((2, f"{long_overlaps} career-role overlap longer than one year"))



        # ── HARD FLAG 5: Impossible startup founding dates ──
        if isinstance(career, list):
            impossible_startups = {
                "Sarvam AI": date(2023, 1, 1),
                "Krutrim": date(2023, 1, 1),
            }
            for role in career:
                if not isinstance(role, dict):
                    continue
                company = role.get("company")
                start = _parse_date(role.get("start_date"))
                if company in impossible_startups and start and start < impossible_startups[company]:
                    flags.append((3, f"Role at {company} started in {start.year} but company was founded in {impossible_startups[company].year}"))

        # ── MEDIUM FLAG 6: All skills uniform expert + 0 months ──
        if isinstance(skills, list) and len(skills) >= 8:
            profs = [s.get("proficiency") for s in skills if isinstance(s, dict)]
            durs = [s.get("duration_months", -1) for s in skills if isinstance(s, dict)]
            if profs and len(set(profs)) == 1 and profs[0] == "expert" and all(d == 0 for d in durs):
                flags.append((3, f"All {len(skills)} skills: expert proficiency, 0 months"))

        # ── MEDIUM FLAG 7: Career sum vastly exceeds claimed experience ──
        if isinstance(career, list):
            total_career_months = sum(
                r.get("duration_months", 0) for r in career if isinstance(r, dict)
            )
            if exp_years > 0:
                ratio = total_career_months / (exp_years * 12)
                if ratio > 3.0:
                    flags.append((2, f"Career total={total_career_months}mo vs claimed {exp_years}yrs ({ratio:.1f}x)"))
                elif ratio > 1.8:
                    flags.append((1, f"Career total={total_career_months}mo vs claimed {exp_years}yrs ({ratio:.1f}x) — mild inflation"))

        # Compute severity
        severity = sum(s for s, _ in flags)
        flag_strings = [desc for _, desc in flags]
        is_honeypot = severity >= self.HONEYPOT_THRESHOLD

        return HoneypotResult(
            candidate_id=cid,
            is_honeypot=is_honeypot,
            severity=severity,
            flags=flag_strings,
        )

    def detect_batch(self, candidates_raw: List[Dict[str, Any]]) -> Dict[str, HoneypotResult]:
        """Detect honeypots across a batch of raw candidate dicts.

        Returns:
            Dict mapping candidate_id -> HoneypotResult
        """
        results = {}
        honeypot_count = 0
        for raw in candidates_raw:
            result = self.detect(raw)
            results[result.candidate_id] = result
            if result.is_honeypot:
                honeypot_count += 1

        logger.info(
            "Honeypot detection complete: %d / %d flagged (threshold=%d)",
            honeypot_count,
            len(candidates_raw),
            self.HONEYPOT_THRESHOLD,
        )
        return results
