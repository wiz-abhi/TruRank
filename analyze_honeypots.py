"""
Refined honeypot detection — targeting ~80 'subtly impossible' profiles.

From the docs:
- "8 years of experience at a company founded 3 years ago"
- "'expert' proficiency in 10 skills with 0 years used"
- Keyword stuffers (all AI keywords but title is Marketing Manager)
- Behavioral impossibilities
"""

import json
import sys
import io
from datetime import datetime, date
from collections import Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

DATA_PATH = r"[PUB] India_runs_data_and_ai_challenge/[PUB] India_runs_data_and_ai_challenge/India_runs_data_and_ai_challenge/candidates.jsonl"

def parse_date(d):
    if not d:
        return None
    try:
        return datetime.strptime(str(d).strip(), "%Y-%m-%d").date()
    except:
        return None

def analyze_candidate(c):
    """Return honeypot flags with severity scores."""
    flags = []
    cid = c["candidate_id"]
    profile = c.get("profile", {})
    skills = c.get("skills", [])
    career = c.get("career_history", [])
    education = c.get("education", [])
    signals = c.get("redrob_signals", {})
    exp_years = profile.get("years_of_experience", 0)
    title = profile.get("current_title", "").lower()

    # === HARD FLAG 1: Expert proficiency with 0 months duration (multiple) ===
    expert_zero = [s for s in skills if s.get("proficiency") == "expert" and s.get("duration_months", -1) == 0]
    if len(expert_zero) >= 3:
        flags.append(("EXPERT_ZERO_DURATION", 3, f"{len(expert_zero)} 'expert' skills with 0 months"))

    # === HARD FLAG 2: Role duration_months vastly exceeds calendar span ===
    for role in career:
        start = parse_date(role.get("start_date"))
        end = parse_date(role.get("end_date"))
        dur_months = role.get("duration_months", 0)
        if start and end:
            actual_months = (end.year - start.year) * 12 + (end.month - start.month)
            if dur_months > 0 and actual_months > 0 and dur_months > actual_months * 2:
                flags.append(("DURATION_VS_CALENDAR", 3, f"{role.get('company','?')}: claims {dur_months}mo, dates span {actual_months}mo"))

    # === HARD FLAG 3: Career history duration exceeds time since start ===
    for role in career:
        start = parse_date(role.get("start_date"))
        dur_months = role.get("duration_months", 0)
        if start:
            months_available = (2026 - start.year) * 12 + (6 - start.month)
            if dur_months > months_available + 3 and dur_months > 24:
                flags.append(("DURATION_EXCEEDS_EXISTENCE", 3, f"{role.get('company','?')}: claims {dur_months}mo but only {months_available}mo since start"))

    # === HARD FLAG 4: Massive experience gap vs education ===
    # Graduated in 2022 but claims 10+ years experience = impossible
    if education:
        latest_grad = max((e.get("end_year", 0) for e in education), default=0)
        if latest_grad >= 2018:
            years_since_grad = 2026 - latest_grad
            if exp_years > years_since_grad * 2 and exp_years > 8:
                flags.append(("IMPOSSIBLE_EXP_VS_GRAD", 3, f"{exp_years}yrs exp but graduated {latest_grad} ({years_since_grad}yrs ago)"))

    # === MEDIUM FLAG 5: Keyword stuffer — non-tech title with tons of AI skills ===
    ai_keywords = {"machine learning", "deep learning", "nlp", "natural language processing",
                   "pytorch", "tensorflow", "keras", "bert", "transformers", "rag",
                   "llm", "gpt", "langchain", "vector database", "faiss", "pinecone",
                   "computer vision", "reinforcement learning", "neural network",
                   "embedding", "hugging face", "huggingface", "scikit-learn",
                   "pandas", "numpy", "spark", "mlops", "model training",
                   "data science", "artificial intelligence", "ai"}
    non_tech_titles = {"marketing manager", "sales executive", "hr manager", "accountant",
                       "content writer", "graphic designer", "customer support",
                       "operations manager", "civil engineer", "mechanical engineer",
                       "project manager", "business analyst", "sales manager",
                       "financial analyst", "recruiter", "admin", "receptionist"}
    
    skill_names_lower = {s.get("name", "").lower() for s in skills}
    ai_skill_count = len(ai_keywords & skill_names_lower)
    is_non_tech = any(nt in title for nt in non_tech_titles)
    
    if is_non_tech and ai_skill_count >= 6:
        flags.append(("KEYWORD_STUFFER", 2, f"Title='{profile.get('current_title','')}' but has {ai_skill_count} AI skills"))

    # === MEDIUM FLAG 6: All skills at same proficiency + same endorsements (synthetic) ===
    if len(skills) >= 8:
        proficiencies = [s.get("proficiency") for s in skills]
        endorsements = [s.get("endorsements", 0) for s in skills]
        if len(set(proficiencies)) == 1 and proficiencies[0] == "expert":
            durations = [s.get("duration_months", -1) for s in skills]
            if all(d == 0 for d in durations):
                flags.append(("UNIFORM_EXPERT_ZERO", 3, f"All {len(skills)} skills: expert, 0 months"))

    # === MEDIUM FLAG 7: End date before start date ===
    for role in career:
        start = parse_date(role.get("start_date"))
        end = parse_date(role.get("end_date"))
        if start and end and end < start:
            flags.append(("END_BEFORE_START", 3, f"{role.get('company','?')}: end={end} < start={start}"))

    # === MEDIUM FLAG 8: Career sum vastly exceeds claimed experience ===
    total_career_months = sum(r.get("duration_months", 0) for r in career)
    if exp_years > 0 and total_career_months > exp_years * 12 * 3:
        flags.append(("CAREER_SUM_IMPOSSIBLE", 2, f"Career total={total_career_months}mo vs claimed {exp_years}yrs ({exp_years*12}mo)"))

    # === MEDIUM FLAG 9: Assessment scores but skills don't match ===
    assess = signals.get("skill_assessment_scores", {})
    if assess:
        assessed_skills = set(k.lower() for k in assess.keys())
        listed_skills = set(s.get("name", "").lower() for s in skills)
        if assessed_skills and listed_skills:
            overlap = assessed_skills & listed_skills
            if len(overlap) == 0 and len(assessed_skills) >= 3:
                flags.append(("ASSESSMENT_SKILL_MISMATCH", 2, f"Assessed {len(assessed_skills)} skills, listed {len(listed_skills)} skills, 0 overlap"))

    # Compute severity score
    severity = sum(s for _, s, _ in flags)
    return flags, severity


def main():
    print("Loading candidates...")
    candidates = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    print(f"Loaded {len(candidates)} candidates.\n")

    scored_candidates = []
    flag_counter = Counter()

    for c in candidates:
        flags, severity = analyze_candidate(c)
        if flags:
            for name, _, _ in flags:
                flag_counter[name] += 1
            scored_candidates.append((c["candidate_id"], severity, flags, c))

    # Sort by severity (highest first)
    scored_candidates.sort(key=lambda x: -x[1])

    print(f"=== FLAG DISTRIBUTION (all candidates) ===")
    for flag, count in flag_counter.most_common():
        print(f"  {flag}: {count}")

    # The challenge says ~80 honeypots. Let's look at severity thresholds.
    for threshold in [6, 5, 4, 3, 2]:
        count = sum(1 for _, sev, _, _ in scored_candidates if sev >= threshold)
        print(f"\n  Severity >= {threshold}: {count} candidates")

    # Use severity >= 3 as our honeypot threshold (hard flags)
    honeypots = [(cid, sev, flags, c) for cid, sev, flags, c in scored_candidates if sev >= 3]
    print(f"\n=== HONEYPOT CANDIDATES (severity >= 3): {len(honeypots)} ===")
    
    for cid, sev, flags, c in honeypots[:30]:
        title = c.get("profile", {}).get("current_title", "?")
        exp = c.get("profile", {}).get("years_of_experience", 0)
        print(f"\n  {cid} (severity={sev}, title='{title}', exp={exp}yrs):")
        for name, s, desc in flags:
            print(f"    [{s}] {name}: {desc}")

    # Save honeypot IDs
    honeypot_ids = set(cid for cid, _, _, _ in honeypots)
    with open("honeypot_ids.txt", "w") as f:
        for cid in sorted(honeypot_ids):
            f.write(cid + "\n")
    print(f"\nSaved {len(honeypot_ids)} honeypot IDs to honeypot_ids.txt")

    # Check our submission
    import csv
    submission_ids = []
    try:
        with open("submission.csv", "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                submission_ids.append(row["candidate_id"])
    except FileNotFoundError:
        print("No submission.csv found.")
        return

    honeypot_in_sub = [cid for cid in submission_ids if cid in honeypot_ids]
    pct = len(honeypot_in_sub) / len(submission_ids) * 100 if submission_ids else 0
    print(f"\n=== SUBMISSION CHECK ===")
    print(f"Honeypots in top 100: {len(honeypot_in_sub)} ({pct:.1f}%)")
    if honeypot_in_sub:
        print(f"IDs: {honeypot_in_sub}")
    print("SAFE" if pct <= 10 else "DANGER: >10% honeypot rate!")


if __name__ == "__main__":
    main()
