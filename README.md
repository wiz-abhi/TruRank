# 🇮🇳 IndiaRanks — Intelligent Candidate Ranking System

### India Runs by Redrob AI — Track 1: Data & AI Challenge

> An AI-powered candidate ranking engine that goes beyond keyword filtering to deeply understand context, predict relevance, and rank candidates using 9 distinct signals — built for India's job market at scale.

---

## What This Builds

Traditional resume screening relies on keyword matching — "does this resume contain Python?" — which fails to capture context, career trajectory, or cultural fit. IndiaRanks replaces this with a **multi-signal ranking pipeline** that combines:

- **Semantic embeddings** (sentence-transformers) for deep language understanding
- **India-specific heuristics** — education tier classification (IIT/NIT/BITS), startup vs enterprise culture mapping, skill synonyms common on Indian resumes
- **Behavioral signals** — career velocity, profile freshness, skill recency
- **Full explainability** — every candidate gets 3 human-readable reasons for their rank + flag notes for edge cases

The result: a ranked candidate list where judges (and recruiters) can instantly see **why** each candidate placed where they did, not just a score.

---

## Architecture

```
[precompute.py]
Candidates (JSONL) → Parser → Embeddings (sentence-transformers) → candidates_cache.pkl

[rank.py]
JD + candidates_cache.pkl → Semantic Similarity → Signal Engine → Ranker → Explainer → submission.csv
```

The system computes 9 orthogonal signals per candidate, weighted into a composite score:

| # | Signal | Weight | What it captures |
|---|--------|--------|-----------------|
| 1 | Semantic Similarity | 0.30 | Deep language match between JD and profile |
| 2 | Skill Match | 0.20 | Hard + soft skill overlap with synonym resolution |
| 3 | Skill Recency | 0.10 | Penalises stale skills (>3 years unused) |
| 4 | Career Velocity | 0.10 | Role progression rate (penalises both stagnation and hopping) |
| 5 | Experience Fit | 0.10 | Closeness to JD's experience requirement |
| 6 | Domain Alignment | 0.08 | Has the candidate worked in the JD's domain? |
| 7 | Profile Freshness | 0.07 | How recently was the candidate active? |
| 8 | Culture Fit Proxy | 0.05 | Startup vs enterprise background alignment |
| 9 | Education Tier Bonus | +0.15 | Additive bonus for IIT/IIM/BITS/NIT (not a gate) |

All weights are configurable in [`config.yaml`](config.yaml).

---

## Tech Stack

| Component | Tool |
|---|---|
| Semantic embeddings | `sentence-transformers` (all-MiniLM-L6-v2 / bge-large-en-v1.5) |
| Signal computation | Python, pandas, NumPy, scikit-learn |
| JD parsing | Regex + keyword heuristics (LLM-optional) |
| Demo UI | Streamlit + Plotly |
| Output | CSV (pandas) |
| Testing | pytest |
| Config | YAML |

---

## How to Run

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/india-runs-track1.git
cd india-runs-track1

python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Run on provided dataset

```bash
python -m src.ranker \
    --jd data/sample/jd.json \
    --profiles data/sample/profiles.csv \
    --output outputs/ranked_output.csv
```

### 3. Run the demo

```bash
streamlit run app/demo.py
```

### 4. Run tests

```bash
pytest tests/ -v
```

### 5. Run benchmark

```bash
python benchmark.py --count 1000
```

---

## Project Structure

```
india_runs_track1/
├── README.md
├── requirements.txt
├── config.yaml                # All weights, thresholds, company lists
├── .gitignore
├── data/
│   ├── raw/                   # Original dataset files
│   ├── processed/             # Cleaned/featurized data
│   └── sample/                # Sample JD + 15 profiles for demo
├── src/
│   ├── __init__.py
│   ├── jd_parser.py           # JD understanding + feature extraction
│   ├── profile_parser.py      # Candidate profile normalization
│   ├── embeddings.py          # Semantic embedding pipeline
│   ├── signals.py             # 9 behavioral + activity signal computation
│   ├── ranker.py              # Scoring + ranking logic + CLI
│   ├── explainer.py           # Natural language explanations per candidate
│   └── utils.py               # Logging, config, helpers
├── tests/
│   └── test_signals.py        # 20+ unit tests for signal computation
├── app/
│   └── demo.py                # Streamlit demo with radar charts
├── outputs/
│   └── ranked_output.csv      # Final submission file
├── benchmark.py               # Performance benchmarking script
└── docs/
    └── architecture.md        # System architecture notes
```

---

## India-Native Thinking

This system is purpose-built for the Indian job market:

- **Education tier awareness**: Automatically classifies 30+ Indian institutions into Tier 1/2/3 — but as a *small additive bonus*, not a gate. This reflects the reality that IIT grads often have strong fundamentals, while explicitly preventing bias.
- **Skill synonym dictionary**: 65+ entries covering abbreviations common on Indian resumes ("ML" → "Machine Learning", "ReactJS" → "React", "K8s" → "Kubernetes").
- **Company culture mapping**: 50 Indian startups (Razorpay, CRED, Meesho…) and 20 enterprises (TCS, Infosys, Wipro…) for culture fit scoring.
- **Indian city recognition**: 20+ cities parsed from JDs including both names for Bangalore/Bengaluru, Gurgaon/Gurugram.
- **Experience parsing**: Handles "3 years", "2+ yrs", "18 months" — common formats in Indian CVs.

---

## Explainability

Every candidate receives 3 human-readable reasons and flag notes:

```
Rank #1: Vikram Reddy (89%)
├── Strong semantic match (82%) — profile language closely mirrors JD requirements.
├── Skills in python, tensorflow, nlp directly match the JD's required stack (75% skill match).
└── 6.5 years of experience — within optimal range for this senior-level role.
    ⚠ Flag: Slightly over-qualified

Rank #5: Sneha Gupta (42%)
├── Low skill overlap (15%) — candidate's stack differs significantly from JD requirements.
├── Only 4.0 years experience — but in a different domain (backend vs data_science).
└── VIT Vellore alumnus — Tier 2 education (+8% bonus).
```

---

## Why This Approach Wins

**Beyond keyword filters**: Traditional ATS systems look for exact string matches. They miss that "built production ML pipelines" is relevant to a "Senior Machine Learning Engineer" role. Our semantic embedding approach captures this context because it encodes *meaning*, not just *words*.

**Multi-signal, not single-metric**: A candidate who has the right skills but hasn't been active in 2 years should rank lower than one who's actively looking. A candidate from a startup background is a better fit for a startup JD than one from an enterprise — but they might have the same keywords. Our 9-signal approach captures these nuances in a way that's transparent, configurable, and explainable.

---

## License

MIT
