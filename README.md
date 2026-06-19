# 🇮🇳 IndiaRanks — Intelligent Candidate Ranking System

### India Runs by Redrob AI — Track 1: Data & AI Challenge

> An AI-powered candidate ranking engine that goes beyond keyword filtering to deeply understand context, predict relevance, filter out honeypot profiles, and rank candidates using a multi-signal pipeline — built for India's job market at scale.

---

## What This Builds

Traditional resume screening relies on keyword matching — "does this resume contain Python?" — which fails to capture context, career trajectory, or cultural fit. IndiaRanks replaces this with a **robust, multi-signal ranking pipeline** that combines:

- **Semantic embeddings** (`sentence-transformers`) for deep language understanding
- **Honeypot Detection** — Explicit filtering of impossible profiles (e.g., 0-month expert skills, timelines exceeding calendar dates)
- **Behavioral signals** — 12 specific Redrob signals including notice period, GitHub score, profile completeness, interview completion rate, and responsiveness
- **India-specific heuristics** — Penalties for services-only experience (per JD disqualifiers), and location preferences
- **Full explainability** — Every candidate receives a dynamically generated, plain-language reasoning paragraph referencing their specific job title, company, matching skills, and behavioral traits

The result: a heavily curated, 100-row `submission.csv` completely free of honeypots, where judges can instantly see exactly why a candidate placed where they did.

---

## Architecture

```text
[precompute.py]
Candidates (JSONL) → Parser → Embeddings (sentence-transformers) → candidates_cache.pkl

[rank.py]
JD + candidates_cache.pkl 
      ↓
Semantic Similarity
      ↓
Signal Engine (12 Redrob Signals)
      ↓
Honeypot Detector (Filters out impossible profiles)
      ↓
Explainer Engine (Generates plain-language reasoning)
      ↓
submission.csv
```

### Signal Computation
The `SignalComputer` extracts behavioral metrics into a massive composite multiplier:
- **Notice Period**: Bonus for sub-30 days, penalty for >90 days
- **Recency**: Bonus for <60 days active, heavy penalty for >365 days
- **Response Rate & Time**: Bonuses for fast responders (<24h) and high response rates
- **GitHub Activity Score**: Bonus for scores >40
- **Profile Completeness**: Penalty for <40 score
- **Trust Metrics**: Bonuses for LinkedIn connectivity and recruiter saves, penalties for lack of verified email/phone
- **Disqualifiers**: A severe 90% penalty is applied to candidates whose entire career history consists exclusively of enterprise consulting services (TCS, Infosys, etc.), as explicitly mandated by the actual JD.

All weights and thresholds are strictly configured in [`config.yaml`](config.yaml).

---

## Tech Stack

| Component | Tool |
|---|---|
| Semantic embeddings | `sentence-transformers` (all-MiniLM-L6-v2) |
| Signal computation | Python, pandas, NumPy |
| Output | CSV |

---

## How to Run

### 1. Clone and install

```bash
git clone https://github.com/wiz-abhi/indiaruns.git
cd indiaruns

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Precompute Embeddings

```bash
python precompute.py
```
*This parses all 100K JSONL candidates and saves embeddings to `data/processed/candidates_cache.pkl`. Note: this takes about 45 minutes on a standard CPU.*

### 3. Rank and Generate Submission

```bash
python rank.py
```
*This loads the cache, compares candidates against the hackathon JD, computes all Redrob signals, runs the Honeypot Detector, generates reasoning, and outputs a 100-row `submission.csv`.*

### 4. Run the UI Demo (Bonus)

```bash
python -m streamlit run app/demo.py --server.fileWatcherType none
```
*A beautiful interactive dashboard to visualize the top candidates, their semantic match, signal breakdown, and generated reasoning.*

---

## Project Structure

```text
india_runs_track1/
├── README.md
├── requirements.txt
├── config.yaml                # All weights, thresholds, JD alignment
├── .gitignore
├── submission_metadata.yaml   # Challenge submission metadata
├── data/
│   ├── raw/                   # Original dataset files
│   └── processed/             # Precomputed embeddings cache
├── src/
│   ├── jd_parser.py           # JD understanding + feature extraction
│   ├── profile_parser.py      # Candidate profile normalization
│   ├── embeddings.py          # Semantic embedding pipeline
│   ├── signals.py             # Redrob behavioral + activity signal computation
│   ├── honeypot_detector.py   # Explicit trap detection logic
│   ├── explainer.py           # Natural language explanations per candidate
│   └── utils.py               # Logging, config, helpers
├── precompute.py              # Step 1: Embedding generation
├── rank.py                    # Step 2: Scoring + filtering + ranking logic
└── submission.csv             # Final generated output
```

---

## Explainability

Every candidate receives highly specific, plain-language reasoning.

**Example output from `submission.csv`:**
> *"Senior AI Engineer at Apple with 5.9 years. Strong skill alignment including sentence-transformers, pinecone, weaviate. Moderate semantic match to the core JD. Candidate is highly active, 30-day notice."*

---

## Why This Approach Wins

**Zero Honeypots**: By implementing a hard-coded set of rules targeting temporal impossibilities, we completely bypass the 10% honeypot disqualifier rule.

**Beyond keyword filters**: Traditional ATS systems look for exact string matches. Our semantic embedding approach captures context because it encodes *meaning*, not just *words*.

**Complete JD Alignment**: A candidate who has the right skills but works at a massive IT services firm, or hasn't responded to recruiters in a year, is heavily penalized. Our comprehensive 12-signal behavioral multiplier perfectly mirrors what human recruiters actually care about.

---

## License

MIT
