from datetime import datetime

from src.jd_parser import JobDescription
from src.profile_parser import ProfileParser
from src.signals import SignalComputer


JD = JobDescription(
    required_skills=["python", "embeddings", "vector database", "ndcg"],
    preferred_skills=["learning-to-rank"],
    min_experience_years=5,
    domain="data_science",
    culture_signals=["startup mindset"],
)


def candidate(title, description, skills, location="Pune", relocate=False):
    return {
        "candidate_id": "CAND_0000001",
        "profile": {
            "anonymized_name": "Test Candidate",
            "headline": title,
            "summary": description,
            "location": location,
            "country": "India",
            "years_of_experience": 6,
            "current_title": title,
        },
        "career_history": [{
            "company": "Product Co",
            "title": title,
            "industry": "Software Product",
            "description": description,
            "duration_months": 48,
        }],
        "education": [],
        "skills": skills,
        "redrob_signals": {
            "last_active_date": datetime.now().strftime("%Y-%m-%d"),
            "open_to_work_flag": True,
            "recruiter_response_rate": 0.6,
            "notice_period_days": 30,
            "willing_to_relocate": relocate,
            "skill_assessment_scores": {"Python": 85},
        },
    }


def supported_skills():
    return [
        {"name": "Python", "proficiency": "advanced", "duration_months": 48, "endorsements": 20},
        {"name": "embeddings", "proficiency": "advanced", "duration_months": 30, "endorsements": 12},
        {"name": "vector database", "proficiency": "advanced", "duration_months": 24, "endorsements": 8},
        {"name": "NDCG", "proficiency": "intermediate", "duration_months": 18, "endorsements": 5},
    ]


def score(raw):
    profile = ProfileParser().parse(raw)
    return SignalComputer().compute_all(profile, JD, semantic_sim=0.65)


def test_production_evidence_beats_keyword_stuffing():
    production = candidate(
        "Senior Machine Learning Engineer",
        "Owned and shipped a production semantic search and recommendation system to real users; designed NDCG benchmarks and A/B experiments.",
        supported_skills(),
    )
    stuffed = candidate(
        "Marketing Manager",
        "Managed brand campaigns and social media.",
        [{"name": x, "proficiency": "expert", "duration_months": 0, "endorsements": 0}
         for x in ("Python", "embeddings", "vector database", "NDCG")],
    )
    assert score(production).composite_score > score(stuffed).composite_score * 2


def test_skill_evidence_uses_duration_and_proficiency():
    good = score(candidate("ML Engineer", "Built production retrieval pipelines.", supported_skills()))
    weak = score(candidate("ML Engineer", "Built production retrieval pipelines.", [
        {"name": s["name"], "proficiency": "beginner", "duration_months": 0, "endorsements": 0}
        for s in supported_skills()
    ]))
    assert good.skill_evidence > weak.skill_evidence


def test_zero_duration_expert_claims_reduce_trust():
    good = score(candidate("ML Engineer", "Built production retrieval pipelines.", supported_skills()))
    contradictory = score(candidate("ML Engineer", "Built production retrieval pipelines.", [
        {"name": s["name"], "proficiency": "expert", "duration_months": 0, "endorsements": 0}
        for s in supported_skills()
    ]))
    assert good.profile_trust > contradictory.profile_trust


def test_location_and_relocation_are_used():
    pune = score(candidate("ML Engineer", "Built search systems.", supported_skills(), "Pune"))
    abroad = score(candidate("ML Engineer", "Built search systems.", supported_skills(), "Toronto"))
    relocating = score(candidate("ML Engineer", "Built search systems.", supported_skills(), "Bengaluru", True))
    assert pune.location_fit > relocating.location_fit > abroad.location_fit
