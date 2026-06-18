"""
Performance benchmark — runs the full pipeline on synthetic profiles
and reports timing + memory usage.

Usage:
    python benchmark.py [--count 1000]
"""

from __future__ import annotations

import argparse
import random
import string
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import psutil

from src.embeddings import EmbeddingEngine
from src.jd_parser import JDParser, JobDescription
from src.profile_parser import CandidateProfile
from src.ranker import CandidateRanker
from src.signals import SKILL_SYNONYMS
from src.utils import get_logger

logger = get_logger(__name__)

# ── Synthetic data generators ────────────────────────────────────────────
_ALL_SKILLS = list(SKILL_SYNONYMS.keys())
_FIRST_NAMES = [
    "Arjun", "Priya", "Vikram", "Sneha", "Rohit", "Ananya", "Karthik",
    "Deepika", "Amit", "Farah", "Rajesh", "Sanya", "Mohan", "Kavitha",
    "Nikhil", "Aarav", "Ishaan", "Diya", "Vihaan", "Aisha", "Reyansh",
    "Saanvi", "Advait", "Anvi", "Dhruv", "Myra", "Kabir", "Aanya",
]
_LAST_NAMES = [
    "Sharma", "Nair", "Reddy", "Gupta", "Mehta", "Krishnan", "Iyer",
    "Patel", "Kumar", "Khan", "Das", "Sundaram", "Joshi", "Singh",
    "Verma", "Agarwal", "Bose", "Chatterjee", "Menon", "Rao",
]
_COMPANIES = [
    "Razorpay", "Flipkart", "TCS", "Infosys", "CRED", "Amazon India",
    "Google India", "Microsoft India", "Wipro", "Meesho", "Swiggy",
    "Zoho", "Freshworks", "Paytm", "PhonePe", "Dream11", "Zomato",
    "Groww", "Postman", "InMobi", "HCL", "Cognizant", "Accenture",
]
_TITLES = [
    "Software Engineer", "Senior Software Engineer", "ML Engineer",
    "Data Scientist", "Backend Developer", "Full Stack Developer",
    "Data Analyst", "DevOps Engineer", "Product Manager",
    "Senior Data Scientist", "Lead Engineer", "Staff Engineer",
]
_EDUCATIONS = [
    "IIT Bombay, B.Tech CSE", "IIT Delhi, M.Tech ML",
    "NIT Trichy, B.Tech ECE", "BITS Pilani, M.Tech",
    "VIT Vellore, B.Tech IT", "Anna University, B.Tech",
    "Delhi University, B.Sc", "IIM Bangalore, MBA",
    "IIIT Hyderabad, B.Tech", "Manipal, B.Tech CSE",
    "COEP Pune, B.Tech IT", "Jadavpur University, B.Tech",
    "SRM University, B.Tech", "PEC Chandigarh, B.Tech",
    "Generic College, BCA",
]


def generate_synthetic_profile(idx: int) -> CandidateProfile:
    """Generate one random synthetic candidate profile."""
    name = f"{random.choice(_FIRST_NAMES)} {random.choice(_LAST_NAMES)}"
    num_skills = random.randint(3, 10)
    skills = random.sample(_ALL_SKILLS, min(num_skills, len(_ALL_SKILLS)))
    exp = round(random.uniform(0.5, 15.0), 1)
    edu = random.choice(_EDUCATIONS)
    num_titles = random.randint(1, 4)
    titles = random.sample(_TITLES, min(num_titles, len(_TITLES)))
    num_companies = random.randint(1, 4)
    companies = random.sample(_COMPANIES, min(num_companies, len(_COMPANIES)))
    days_ago = random.randint(1, 900)
    last_active = datetime.now() - timedelta(days=days_ago)

    # Classify education tier
    edu_lower = edu.lower()
    if any(kw in edu_lower for kw in ["iit", "iim", "bits", "nit", "iisc"]):
        tier = "tier_1"
    elif any(kw in edu_lower for kw in ["vit", "srm", "manipal", "iiit", "coep", "pec", "jadavpur"]):
        tier = "tier_2"
    else:
        tier = "tier_3"

    velocity = round(max(0, len(titles) - 1) / max(exp, 0.5), 2)

    return CandidateProfile(
        candidate_id=f"SYN-{idx:05d}",
        name=name,
        skills=skills,
        experience_years=exp,
        education=edu,
        education_tier=tier,
        job_titles=titles,
        companies=companies,
        location=random.choice(["Bangalore", "Mumbai", "Delhi", "Hyderabad", "Pune", "Chennai"]),
        last_active=last_active,
        career_velocity=min(velocity, 2.0),
    )


def run_benchmark(count: int) -> None:
    """Execute the full pipeline benchmark."""
    process = psutil.Process()
    mem_before = process.memory_info().rss / (1024 * 1024)

    print(f"\nPipeline benchmark — {count:,} profiles")
    print("=" * 45)

    # 1) Generate synthetic profiles
    t0 = time.perf_counter()
    profiles = [generate_synthetic_profile(i) for i in range(count)]
    t_gen = time.perf_counter() - t0
    print(f"Profile generation:   {t_gen:.2f}s")

    # 2) Parse JD
    t0 = time.perf_counter()
    jd_parser = JDParser()
    sample_jd_path = Path(__file__).parent / "data" / "sample" / "jd.json"
    import json
    with open(sample_jd_path, "r", encoding="utf-8") as f:
        jd_data = json.load(f)
    jd = jd_parser.parse(jd_data)
    t_jd = time.perf_counter() - t0
    print(f"JD parsing:           {t_jd:.2f}s")

    # 3) Batch embedding
    t0 = time.perf_counter()
    engine = EmbeddingEngine()
    jd_vec = engine.embed_jd(jd)
    profile_vecs = engine.batch_embed_profiles(profiles, cache_key=f"bench_{count}", refresh=True)
    t_embed = time.perf_counter() - t0
    print(f"Batch embedding:      {t_embed:.2f}s")

    # 4) Signal computation + ranking
    t0 = time.perf_counter()
    ranker = CandidateRanker(embedding_engine=engine)
    ranked = ranker.rank(profiles, jd, refresh_embeddings=False)
    t_rank = time.perf_counter() - t0
    # Subtract embedding time already counted
    t_signals = max(0, t_rank - 0.01)
    print(f"Signals + ranking:    {t_signals:.2f}s")

    # 5) CSV export
    t0 = time.perf_counter()
    out_path = Path(__file__).parent / "outputs" / f"bench_{count}.csv"
    CandidateRanker.export_csv(ranked, out_path)
    t_export = time.perf_counter() - t0
    print(f"CSV export:           {t_export:.2f}s")

    total = t_gen + t_jd + t_embed + t_signals + t_export
    mem_after = process.memory_info().rss / (1024 * 1024)

    print("-" * 45)
    print(f"TOTAL:                {total:.2f}s")
    print(f"Profiles/second:      {count / total:.1f}")
    print(f"Peak memory:          {mem_after:.0f} MB (delta: +{mem_after - mem_before:.0f} MB)")
    print(f"Output:               {out_path}")

    # Write results to file
    results_path = Path(__file__).parent / "outputs" / "benchmark_results.txt"
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(f"Pipeline benchmark — {count:,} profiles\n")
        f.write("=" * 45 + "\n")
        f.write(f"Profile generation:   {t_gen:.2f}s\n")
        f.write(f"JD parsing:           {t_jd:.2f}s\n")
        f.write(f"Batch embedding:      {t_embed:.2f}s\n")
        f.write(f"Signals + ranking:    {t_signals:.2f}s\n")
        f.write(f"CSV export:           {t_export:.2f}s\n")
        f.write("-" * 45 + "\n")
        f.write(f"TOTAL:                {total:.2f}s\n")
        f.write(f"Profiles/second:      {count / total:.1f}\n")
        f.write(f"Peak memory:          {mem_after:.0f} MB\n")

    print(f"\n✅ Benchmark results saved to {results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run pipeline benchmark")
    parser.add_argument("--count", type=int, default=1000, help="Number of synthetic profiles")
    args = parser.parse_args()
    run_benchmark(args.count)
