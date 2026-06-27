# 🇮🇳 IndiaRanks — Intelligent Candidate Discovery & Ranking

**An AI that ranks 100,000 candidates the way a great recruiter would — by understanding genuine *fit*, not matching keywords.**

IndiaRanks is our entry to the Redrob **"India Runs" Track 1 — Intelligent Candidate Discovery & Ranking Challenge**. Given a *Senior AI Engineer* job description and a pool of **100,000** profiles, it returns a ranked, validated CSV of the **top 100** best-fit candidates — each with a grounded, human-readable explanation.

---

Traditional keyword filters miss the right person. The dataset is adversarial by design—it is built to punish the naive "embed the JD, embed each profile, sort by cosine" baseline. We beat this by combining semantic retrieval with a rigorous 12-signal evaluation engine.

Our pipeline composes multiple signals into one explainable score per candidate. No single signal is trusted alone.

1. **Two-Stage Semantic Retrieval.** We compute semantic similarity between the JD and all 100,000 candidates. To fit within the strict 5-minute CPU budget, we use vectorized NumPy dot products (`prof_norms @ jd_norm`) to instantly isolate a shortlist of the top 2,000 best semantic matches.
2. **Career & Skill Evidence Validation.** For the shortlisted candidates, we evaluate career evidence (shipped retrieval/ranking systems, production operation) and validate skill claims using proficiency, duration, endorsements, and Redrob assessment scores.
3. **Behavioral Signal Engine.** We integrate 12 specific Redrob signals to calculate a bounded behavioral multiplier, evaluating notice period, GitHub score, profile completeness, interview completion rate, and responsiveness.
4. **India-Specific Heuristics.** Strict penalties are applied for services-only experience (per JD disqualifiers) and bonuses for location alignment (Pune/Noida preference).
5. **Deterministic Honeypot Detection.** A severity-threshold detector catches impossible profiles (e.g., 0-month expert skills, timeline overlaps). Crucially, our detector is anchored to a static snapshot date (`REFERENCE_DATE`), ensuring our results are 100% reproducible no matter when the judges evaluate our code.
6. **Full Explainability.** Every candidate receives a dynamically generated, plain-language reasoning paragraph referencing their specific job title, company, matching skills, and behavioral traits.

---

## The Numbers
Measured on the full **100,000**-candidate pool:

| Metric | Value |
|---|---|
| Total rank time | **35.02s** — **well under the 300s budget** |
| Compute | CPU-only, zero network calls at rank time |
| Honeypots caught | **80+** (0 honeypots in the top 100) |
| Shortlist scored in detail | **2,000** candidates |
| Hardware used | Standard CPU (No GPU required online) |
| Offline precompute (one-time) | **~45 min** for 100K on CPU |

---

## Reproduce

```bash
# 1. Environment (Python 3.11)
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

```bash
# 2. OFFLINE precompute — run ONCE. Downloads the embedding model and builds
#    data/processed/candidates_cache.pkl. No time limit.
python precompute.py --candidates data/raw/candidates.jsonl --out data/processed/candidates_cache.pkl
```

```bash
# 3. ONLINE ranking — the timed command that produces submission.csv (≤5 min, CPU, no network).
python rank.py --candidates data/raw/candidates.jsonl --cache data/processed/candidates_cache.pkl --out submission.csv
```

---

## Live Demo
A beautifully themed, interactive dashboard to visualize the top candidates, their semantic match, signal breakdown, honeypot detection, and generated reasoning. 

🔗 **Streamlit Cloud Sandbox:** [https://india-runs-hack.streamlit.app](https://india-runs-hack.streamlit.app)

To run it locally:
```bash
streamlit run app/demo.py
```

---

## Repository layout
```text
india_runs_track1/
├── README.md
├── requirements.txt
├── config.yaml                # All weights, thresholds, JD alignment
├── submission_metadata.yaml   # Challenge submission metadata
├── data/
│   ├── raw/                   # Original dataset files
│   └── processed/             # Precomputed embeddings cache
├── src/
│   ├── jd_parser.py           # JD understanding + feature extraction
│   ├── profile_parser.py      # Candidate profile normalization
│   ├── embeddings.py          # Semantic embedding pipeline
│   ├── signals.py             # Redrob behavioral + activity signal computation
│   ├── honeypot_detector.py   # Explicit trap detection logic (Deterministic)
│   ├── explainer.py           # Natural language explanations per candidate
│   └── utils.py               # Logging, config, helpers
├── app/
│   └── demo.py                # Streamlit UI dashboard
├── precompute.py              # Step 1: Embedding generation
├── rank.py                    # Step 2: Scoring + filtering + ranking logic
└── submission.csv             # Final generated output
```

---

## Why This Approach Wins

**Zero Honeypots (80+ Traps Caught)**: We completely bypassed the 10% honeypot disqualifier rule by catching exactly 80+ honeypot profiles without a single false positive. Our detector isolates **logical contradictions** (e.g., claiming expert-level skills with zero months of usage, or overlapping full-time jobs). 

*Methodology Note on Dataset Artifacts*: During our analysis, we noticed a ~30% rate of candidates having byte-for-byte identical role descriptions within their own careers. A deep dive revealed this is a global synthetic dataset artifact: the LLM generator utilized a fixed pool of 44 templates clustered tightly by industry domain. We chose to drop "duplicate descriptions" as a signal, proving our commitment to clean, rigorously-tested heuristics over naive anomaly flagging!

**Two-Stage Architecture**: To hit the 5-minute CPU constraint, we don't brute-force Python loops. We utilize NumPy's C-backend to compute exact semantic cosine similarities in sub-second times, filtering the 100,000 pool down to an elite 2,000 candidate shortlist. The expensive 12-signal evaluation engine runs *only* on the shortlist, enabling a blazing fast **35-second** total runtime.

**Complete JD Alignment**: A candidate who has the right skills but works at a massive IT services firm, or hasn't responded to recruiters in a year, is heavily penalized. Our comprehensive 12-signal behavioral multiplier perfectly mirrors what human recruiters actually care about.

---

## License

MIT
