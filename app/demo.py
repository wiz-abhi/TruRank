"""
Streamlit demo — IndiaRanks · Intelligent Candidate Ranking System
India Runs by Redrob AI · Track 1: Data & AI Challenge

Run with:
    streamlit run app/demo.py
"""

from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from sentence_transformers import SentenceTransformer

from src.jd_parser import JobDescription
from src.profile_parser import ProfileParser, CandidateProfile
from src.signals import SignalComputer, SignalScores
from src.explainer import ExplainerEngine
from src.honeypot_detector import HoneypotDetector
from src.utils import load_config

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IndiaRanks · Intelligent Candidate Ranking",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Premium CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Global dark background ── */
.stApp {
    background: linear-gradient(160deg, #09090b 0%, #0c0c0e 30%, #111113 60%, #09090b 100%);
}

/* ── Animated hero ── */
.hero-container {
    text-align: center;
    padding: 2.5rem 1rem 1rem;
    position: relative;
}
.hero-title {
    font-size: 3.2rem;
    font-weight: 900;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #ffffff 0%, #e8c547 35%, #ffffff 65%, #e8c547 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 4s linear infinite;
    margin-bottom: 0.2rem;
}
@keyframes shimmer {
    0% { background-position: 0% center; }
    100% { background-position: 200% center; }
}
.hero-subtitle {
    color: rgba(200, 200, 210, 0.85);
    font-size: 1.05rem;
    font-weight: 400;
    letter-spacing: 0.3px;
}
.hero-badge {
    display: inline-block;
    background: rgba(232, 197, 71, 0.08);
    border: 1px solid rgba(232, 197, 71, 0.25);
    color: #e8c547;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 4px 14px;
    border-radius: 20px;
    margin-top: 0.8rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

/* ── Glassmorphism metric cards ── */
.glass-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 18px;
    padding: 1.3rem 1.5rem;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}
.glass-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(232,197,71,0.5), transparent);
    opacity: 0;
    transition: opacity 0.3s;
}
.glass-card:hover {
    transform: translateY(-4px);
    border-color: rgba(232, 197, 71, 0.2);
    box-shadow: 0 12px 40px rgba(232, 197, 71, 0.06), 0 4px 12px rgba(0,0,0,0.4);
}
.glass-card:hover::before { opacity: 1; }
.glass-value {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(135deg, #ffffff, #e0e0e0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.glass-label {
    font-size: 0.72rem;
    color: rgba(160, 160, 165, 0.9);
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-weight: 600;
    margin-top: 2px;
}
.glass-icon {
    font-size: 1.6rem;
    margin-bottom: 4px;
}

/* ── Candidate cards ── */
.cand-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.5rem;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
}
.cand-card:hover {
    background: rgba(255,255,255,0.05);
    border-color: rgba(232, 197, 71, 0.2);
    box-shadow: 0 6px 24px rgba(0,0,0,0.3);
}
.cand-card.top-3 {
    border-left: 3px solid;
}
.cand-card.rank-1-card { border-left-color: #e8c547; }
.cand-card.rank-2-card { border-left-color: #a0a0a5; }
.cand-card.rank-3-card { border-left-color: #c78a50; }

/* ── Honeypot card ── */
.hp-card {
    background: rgba(255, 40, 40, 0.04);
    border: 1px solid rgba(255, 80, 80, 0.18);
    border-radius: 16px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.5rem;
    border-left: 3px solid rgba(255, 80, 80, 0.5);
}

/* ── Rank badges ── */
.rank-circle {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 40px;
    height: 40px;
    border-radius: 50%;
    font-weight: 800;
    font-size: 0.9rem;
    flex-shrink: 0;
}
.rank-1 { background: linear-gradient(135deg, #e8c547, #d4a017); color: #111; }
.rank-2 { background: linear-gradient(135deg, #b0b0b5, #808085); color: #111; }
.rank-3 { background: linear-gradient(135deg, #c78a50, #a06030); color: #fff; }
.rank-n { background: rgba(255,255,255,0.06); color: #ccc; border: 1px solid rgba(255,255,255,0.1); }

/* ── Score bar ── */
.score-track {
    background: rgba(255,255,255,0.06);
    border-radius: 8px;
    height: 10px;
    overflow: hidden;
    flex: 1;
}
.score-fill {
    height: 100%;
    border-radius: 8px;
    background: linear-gradient(90deg, #e8c547, #d4a017, #e8c547);
    transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ── Tags ── */
.tag {
    display: inline-block;
    font-size: 0.7rem;
    padding: 3px 10px;
    border-radius: 20px;
    margin: 2px 4px 2px 0;
    font-weight: 500;
}
.tag-skill { background: rgba(232,197,71,0.1); color: #e8c547; }
.tag-warn { background: rgba(255,107,107,0.1); color: #ff8888; }
.tag-hp { background: rgba(255,50,50,0.12); color: #ff5555; font-weight: 600; }
.tag-good { background: rgba(67,233,123,0.1); color: #6fd89c; }

/* ── Signal mini-bars ── */
.sig-row { display: flex; align-items: center; margin-bottom: 5px; gap: 8px; }
.sig-name { width: 120px; font-size: 0.72rem; color: #8888aa; font-weight: 500; }
.sig-track { flex:1; background: rgba(255,255,255,0.06); border-radius: 4px; height: 6px; overflow: hidden; }
.sig-fill { height: 100%; border-radius: 4px; }
.sig-val { width: 38px; text-align: right; font-size: 0.72rem; color: #bbb; font-weight: 600; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(9,9,11,0.98) !important;
    border-right: 1px solid rgba(255,255,255,0.05);
}
section[data-testid="stSidebar"] .stMarkdown h3 {
    background: linear-gradient(135deg, #ffffff, #e8c547);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: rgba(255,255,255,0.02);
    border-radius: 12px;
    padding: 4px;
    border: 1px solid rgba(255,255,255,0.06);
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    font-weight: 600;
    font-size: 0.85rem;
    padding: 8px 20px;
}
.stTabs [aria-selected="true"] {
    background: rgba(232,197,71,0.1) !important;
}

/* ── Info panels ── */
.info-panel {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 1.5rem;
}
.info-panel h4 { color: #e8c547; margin-bottom: 0.5rem; font-size: 1rem; }
.info-panel p { color: rgba(200,200,205,0.85); font-size: 0.85rem; line-height: 1.7; }

/* ── Divider ── */
.glow-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(232,197,71,0.2), transparent);
    margin: 1.5rem 0;
    border: none;
}

/* ── Pipeline steps ── */
.pipe-step {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px;
    padding: 1rem 1.2rem;
    text-align: center;
    transition: all 0.3s;
}
.pipe-step:hover {
    border-color: rgba(232,197,71,0.2);
    background: rgba(255,255,255,0.05);
}
.pipe-num {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 30px; height: 30px;
    border-radius: 50%;
    background: linear-gradient(135deg, #e8c547, #d4a017);
    color: #111;
    font-weight: 700;
    font-size: 0.8rem;
    margin-bottom: 6px;
}
.pipe-label { color: #fff; font-size: 0.82rem; font-weight: 600; }
.pipe-desc { color: #8888aa; font-size: 0.7rem; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)

# ── Helper functions ─────────────────────────────────────────────────────

def _rank_badge(rank: int) -> str:
    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    cls = f"rank-{rank}" if rank <= 3 else "rank-n"
    label = medals.get(rank, str(rank))
    return f'<span class="rank-circle {cls}">{label}</span>'


def _score_bar(pct: float) -> str:
    return f"""
    <div style="display:flex;align-items:center;gap:10px;">
        <div class="score-track"><div class="score-fill" style="width:{pct:.0f}%"></div></div>
        <span style="color:#fff;font-weight:700;font-size:0.85rem;min-width:42px;text-align:right;">{pct:.0f}%</span>
    </div>
    """


def _signal_bar(label: str, value: float, color: str) -> str:
    pct = max(0, min(100, value * 100))
    return f"""
    <div class="sig-row">
        <span class="sig-name">{label}</span>
        <div class="sig-track"><div class="sig-fill" style="width:{pct:.0f}%;background:{color};"></div></div>
        <span class="sig-val">{value:.2f}</span>
    </div>
    """


SIGNAL_CONFIG = {
    "career_evidence":     {"label": "Career Evidence",   "color": "#e8c547"},
    "semantic_similarity": {"label": "Semantic Match",    "color": "#ffffff"},
    "skill_evidence":      {"label": "Skill Evidence",    "color": "#d4a017"},
    "skill_match":         {"label": "Skill Match",       "color": "#c0c0c5"},
    "experience_fit":      {"label": "Experience Fit",    "color": "#6fd89c"},
    "career_stability":    {"label": "Career Stability",  "color": "#e8c547"},
    "product_company_fit": {"label": "Product Co. Fit",   "color": "#c78a50"},
    "domain_alignment":    {"label": "Domain Alignment",  "color": "#b0b0b5"},
    "location_fit":        {"label": "Location Fit",      "color": "#d4a017"},
    "culture_fit":         {"label": "Culture Fit",       "color": "#e0d5b0"},
    "skill_recency":       {"label": "Skill Depth",       "color": "#c0c0c5"},
    "work_mode_fit":       {"label": "Work Mode Fit",     "color": "#6fd89c"},
    "profile_trust":       {"label": "Profile Trust",     "color": "#ff6b6b"},
}


# ── Cached model loading ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# ── SIDEBAR ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🇮🇳 IndiaRanks")
    st.markdown(
        '<span style="color:#8888aa;font-size:0.78rem;font-weight:500;">'
        'Evidence-First Candidate Ranking Engine</span>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    st.markdown("**📂 Input Data**")
    data_source = st.radio(
        "Choose data source:",
        ["Use Sample Data (data/raw/sample_candidates.json)", "Upload Candidates JSONL/JSON"],
        label_visibility="collapsed"
    )

    uploaded = None
    if data_source == "Upload Candidates JSONL/JSON":
        uploaded = st.file_uploader(
            "JSONL or JSON file",
            type=["jsonl", "json"],
            key="candidates_upload",
            label_visibility="collapsed",
        )

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    st.markdown("**⚙️ Settings**")
    num_show = st.slider("Candidates to display", 5, 100, 15, key="num_show")
    show_signals = st.toggle("Signal breakdowns", value=True, key="show_signals")
    show_honeypots = st.toggle("Honeypot forensics", value=True, key="show_honeypots")

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    run_btn = st.button("🚀 Run Pipeline", use_container_width=True, type="primary")

    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="text-align:center;color:#555;font-size:0.65rem;margin-top:1rem;">'
        'Team Slytherin · Redrob Hackathon 2025<br>'
        'Abhishek Gupta · Sakshi Singh</div>',
        unsafe_allow_html=True,
    )


# ── HERO ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <div class="hero-title">IndiaRanks</div>
    <div class="hero-subtitle">
        Semantic AI · 12-Signal Scoring · Honeypot Detection · Behavioral Intelligence
    </div>
    <div class="hero-badge">Redrob Hackathon · Track 1 · Data & AI</div>
</div>
""", unsafe_allow_html=True)


# ── MAIN PIPELINE ────────────────────────────────────────────────────────
if run_btn:
    raw_candidates = []
    
    if data_source == "Upload Candidates JSONL/JSON":
        if uploaded is None:
            st.error("⚠️ Please upload a JSONL file first, or select 'Use Sample Data'.")
            st.stop()
        content = uploaded.read().decode("utf-8")
        if uploaded.name.endswith(".jsonl"):
            for line in content.strip().split("\n"):
                if line.strip():
                    raw_candidates.append(json.loads(line))
        else:
            data = json.loads(content)
            raw_candidates = data if isinstance(data, list) else [data]
    else:
        sample_path = Path("data/raw/sample_candidates.json")
        if not sample_path.exists():
            st.error(f"Sample data not found at {sample_path}")
            st.stop()
        with open(sample_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            raw_candidates = data if isinstance(data, list) else [data]

    if not raw_candidates:
        st.error("No candidates found.")
        st.stop()

    with st.spinner("⚡ Running the full ranking pipeline..."):
        cfg = load_config()
        target_jd_cfg = cfg.get("target_jd", {})
        jd = JobDescription(
            raw_text="Target Hackathon JD",
            required_skills=target_jd_cfg.get("required_skills", []),
            preferred_skills=target_jd_cfg.get("preferred_skills", []),
            min_experience_years=target_jd_cfg.get("min_experience", 5),
            domain=target_jd_cfg.get("domain", "data_science"),
            culture_signals=target_jd_cfg.get("culture_signals", []),
        )

        parser = ProfileParser()
        profiles = parser.parse_many(raw_candidates)
        detector = HoneypotDetector()

        model = load_model()
        jd_embedding_text = (
            "Senior applied AI engineer who has shipped production retrieval, ranking, "
            "recommendation or search systems to real users. Evidence of evaluation with "
            "NDCG, MRR, MAP or A/B tests; strong Python and product ownership. Skills: "
            + ", ".join(jd.required_skills)
        )
        jd_emb = model.encode(jd_embedding_text, normalize_embeddings=True)
        profile_texts = [p.to_embedding_text() for p in profiles]
        profile_embs = model.encode(profile_texts, normalize_embeddings=True, show_progress_bar=False)

        jd_norm = np.asarray(jd_emb, dtype=np.float32)
        prof_norms = np.asarray(profile_embs, dtype=np.float32)
        semantic_scores = prof_norms @ jd_norm

        computer = SignalComputer()
        explainer = ExplainerEngine()

        scored = []
        honeypots_detected = []
        all_scores_list = []

        for i, profile in enumerate(profiles):
            sim = float(semantic_scores[i])
            scores = computer.compute_all(profile, jd, sim)
            hp_result = detector.detect(profile.raw)
            all_scores_list.append(scores.composite_score)

            if hp_result.is_honeypot:
                honeypots_detected.append((profile, hp_result, scores))
            else:
                reasoning = explainer.explain_rank(profile, scores, jd)
                scored.append((scores.composite_score, profile, scores, reasoning))

        scored.sort(key=lambda x: (-x[0], x[1].candidate_id))
        top_n = scored[:num_show]

    # Store results in session state for tab persistence
    st.session_state["results"] = {
        "profiles": profiles,
        "scored": scored,
        "honeypots": honeypots_detected,
        "top_n": top_n,
        "jd": jd,
        "all_scores": all_scores_list,
        "num_show": num_show,
        "show_signals": show_signals,
        "show_honeypots": show_honeypots,
    }


# ── RENDER RESULTS ───────────────────────────────────────────────────────
if "results" in st.session_state:
    r = st.session_state["results"]
    profiles = r["profiles"]
    scored = r["scored"]
    honeypots_detected = r["honeypots"]
    top_n = r["top_n"]
    jd = r["jd"]
    all_scores = r["all_scores"]
    num_show = r["num_show"]
    show_signals = r["show_signals"]
    show_honeypots = r["show_honeypots"]

    # ── KPI strip ────────────────────────────────────────────────────
    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""
        <div class="glass-card">
            <div class="glass-icon">👥</div>
            <div class="glass-value">{len(profiles)}</div>
            <div class="glass-label">Candidates Analyzed</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="glass-card">
            <div class="glass-icon">✅</div>
            <div class="glass-value">{len(scored)}</div>
            <div class="glass-label">Valid Candidates</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="glass-card">
            <div class="glass-icon">🍯</div>
            <div class="glass-value">{len(honeypots_detected)}</div>
            <div class="glass-label">Honeypots Caught</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        top_score = top_n[0][0] if top_n else 0
        st.markdown(f"""
        <div class="glass-card">
            <div class="glass-icon">🏆</div>
            <div class="glass-value">{top_score:.4f}</div>
            <div class="glass-label">Top Score</div>
        </div>""", unsafe_allow_html=True)
    with c5:
        hp_rate = (len(honeypots_detected) / max(len(profiles), 1)) * 100
        color = "#43e97b" if hp_rate < 10 else "#ff6b6b"
        st.markdown(f"""
        <div class="glass-card">
            <div class="glass-icon">🛡️</div>
            <div class="glass-value" style="-webkit-text-fill-color:{color};">{hp_rate:.1f}%</div>
            <div class="glass-label">Honeypot Rate</div>
        </div>""", unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────
    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    tab_ranking, tab_analytics, tab_honeypots, tab_methodology = st.tabs([
        "🏆 Rankings", "📊 Analytics", "🍯 Honeypot Forensics", "🔬 Methodology"
    ])

    # ════════════════════════════════════════════════════════════════
    # TAB 1: RANKINGS
    # ════════════════════════════════════════════════════════════════
    with tab_ranking:
        st.markdown(f"### Top {len(top_n)} Candidates")

        # JD summary bar
        with st.expander("📋 Target Job Description", expanded=False):
            jd_c1, jd_c2, jd_c3 = st.columns(3)
            with jd_c1:
                st.markdown("**Required Skills**")
                skills_html = " ".join(f'<span class="tag tag-skill">{s}</span>' for s in jd.required_skills[:12])
                st.markdown(skills_html, unsafe_allow_html=True)
            with jd_c2:
                st.markdown("**Preferred Skills**")
                pref_html = " ".join(f'<span class="tag tag-good">{s}</span>' for s in jd.preferred_skills)
                st.markdown(pref_html, unsafe_allow_html=True)
            with jd_c3:
                st.markdown(f"**Experience:** {jd.min_experience_years:.0f}+ years")
                st.markdown(f"**Domain:** {jd.domain.replace('_', ' ').title()}")

        max_score = top_n[0][0] if top_n else 1.0

        for rank_idx, (score, profile, scores, reasoning) in enumerate(top_n, start=1):
            badge = _rank_badge(rank_idx)
            pct = (score / max_score * 100) if max_score > 0 else 0
            bar = _score_bar(pct)

            title = profile.job_titles[0] if profile.job_titles else "Professional"
            company = profile.companies[0] if profile.companies else ""
            loc = profile.location or ""
            loc_html = f' · 📍 {loc}' if loc else ''

            # Extra card class for top 3
            extra_cls = ""
            if rank_idx <= 3:
                extra_cls = f"top-3 rank-{rank_idx}-card"

            # Warning tags
            warns = ""
            if scores.profile_trust < 0.7:
                warns += '<span class="tag tag-warn">⚠ Low Trust</span>'
            if scores.career_stability < 0.4:
                warns += '<span class="tag tag-warn">⚠ Unstable Career</span>'
            if scores.behavioral_multiplier < 0.8:
                warns += '<span class="tag tag-warn">⚠ Low Activity</span>'

            # Positive tags
            goods = ""
            if scores.career_evidence > 0.7:
                goods += '<span class="tag tag-good">✦ Strong Evidence</span>'
            if scores.behavioral_multiplier >= 1.1:
                goods += '<span class="tag tag-good">✦ Highly Active</span>'
            if scores.product_company_fit > 0.7:
                goods += '<span class="tag tag-good">✦ Product Company</span>'

            reasoning_clean = reasoning.replace("\n", " ").strip()

            st.markdown(f"""
            <div class="cand-card {extra_cls}">
                <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
                    {badge}
                    <div style="flex:1;">
                        <div style="color:#fff;font-weight:700;font-size:1.05rem;">{title} at {company}</div>
                        <div style="color:#7777aa;font-size:0.78rem;">
                            {profile.candidate_id} · {profile.experience_years:.1f} yrs exp{loc_html}
                        </div>
                    </div>
                    <div style="width:200px;">{bar}</div>
                </div>
                <div style="color:#b0b0d0;font-size:0.8rem;margin-bottom:6px;line-height:1.5;">{reasoning_clean}</div>
                <div>{goods}{warns}</div>
            </div>
            """, unsafe_allow_html=True)

            if show_signals:
                with st.expander(f"📊 Signal Breakdown — {profile.candidate_id}", expanded=False):
                    sig_col, radar_col = st.columns([1, 1])

                    with sig_col:
                        scores_dict = scores.to_dict()
                        signal_html = ""
                        for key, cfg_item in SIGNAL_CONFIG.items():
                            val = scores_dict.get(key, 0)
                            signal_html += _signal_bar(cfg_item["label"], val, cfg_item["color"])

                        bm = scores_dict.get("behavioral_multiplier", 1.0)
                        bm_color = "#43e97b" if bm >= 1.0 else "#ff6b6b"
                        signal_html += f"""
                        <div style="margin-top:10px;padding-top:8px;border-top:1px solid rgba(255,255,255,0.08);">
                            <div class="sig-row">
                                <span class="sig-name" style="font-weight:700;">Behavioral ×</span>
                                <div class="sig-track"><div class="sig-fill" style="width:{int(bm/1.25*100)}%;background:{bm_color};"></div></div>
                                <span class="sig-val" style="font-weight:700;">{bm:.2f}×</span>
                            </div>
                        </div>
                        """
                        st.markdown(signal_html, unsafe_allow_html=True)

                    with radar_col:
                        labels = [v["label"] for v in SIGNAL_CONFIG.values()]
                        values = [scores_dict.get(k, 0) for k in SIGNAL_CONFIG.keys()]
                        values_closed = values + [values[0]]
                        labels_closed = labels + [labels[0]]

                        fig = go.Figure()
                        fig.add_trace(go.Scatterpolar(
                            r=values_closed, theta=labels_closed,
                            fill="toself",
                            fillcolor="rgba(232, 197, 71, 0.12)",
                            line=dict(color="#e8c547", width=2),
                            marker=dict(size=5, color="#d4a017"),
                        ))
                        fig.update_layout(
                            polar=dict(
                                bgcolor="rgba(0,0,0,0)",
                                radialaxis=dict(visible=True, range=[0, 1],
                                    tickfont=dict(size=8, color="#555"),
                                    gridcolor="rgba(255,255,255,0.06)"),
                                angularaxis=dict(tickfont=dict(size=9, color="#8888aa"),
                                    gridcolor="rgba(255,255,255,0.06)"),
                            ),
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            margin=dict(l=50, r=50, t=15, b=15),
                            height=300,
                            showlegend=False,
                        )
                        st.plotly_chart(fig, use_container_width=True)

    # ════════════════════════════════════════════════════════════════
    # TAB 2: ANALYTICS
    # ════════════════════════════════════════════════════════════════
    with tab_analytics:
        st.markdown("### 📊 Score Distribution & Signal Analytics")

        # Score distribution histogram
        fig_dist = go.Figure()
        valid_scores = [s for s, _, _, _ in scored]
        fig_dist.add_trace(go.Histogram(
            x=valid_scores, nbinsx=40,
            marker=dict(
                color="rgba(232, 197, 71, 0.6)",
                line=dict(color="#e8c547", width=1),
            ),
            name="Valid Candidates",
        ))
        if honeypots_detected:
            hp_scores = [s.composite_score for _, _, s in honeypots_detected]
            fig_dist.add_trace(go.Histogram(
                x=hp_scores, nbinsx=20,
                marker=dict(
                    color="rgba(255, 80, 80, 0.5)",
                    line=dict(color="#ff5555", width=1),
                ),
                name="Honeypots",
            ))
        fig_dist.update_layout(
            title=dict(text="Composite Score Distribution", font=dict(color="#ccc", size=14)),
            xaxis=dict(title="Score", color="#888", gridcolor="rgba(255,255,255,0.04)"),
            yaxis=dict(title="Count", color="#888", gridcolor="rgba(255,255,255,0.04)"),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            barmode="overlay",
            legend=dict(font=dict(color="#aaa")),
            height=350,
            margin=dict(l=50, r=30, t=50, b=40),
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        # Top 10 signal averages
        st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
        st.markdown("#### Average Signal Strength — Top 10 vs All Valid")

        top10 = scored[:10]
        if top10:
            avg_top10 = {}
            avg_all = {}
            for key in SIGNAL_CONFIG:
                avg_top10[key] = np.mean([s.to_dict().get(key, 0) for _, _, s, _ in top10])
                avg_all[key] = np.mean([s.to_dict().get(key, 0) for _, _, s, _ in scored])

            labels = [SIGNAL_CONFIG[k]["label"] for k in SIGNAL_CONFIG]
            top_vals = [avg_top10[k] for k in SIGNAL_CONFIG]
            all_vals = [avg_all[k] for k in SIGNAL_CONFIG]

            fig_compare = go.Figure()
            fig_compare.add_trace(go.Scatterpolar(
                r=top_vals + [top_vals[0]], theta=labels + [labels[0]],
                fill="toself", fillcolor="rgba(232,197,71,0.15)",
                line=dict(color="#e8c547", width=2.5), name="Top 10",
            ))
            fig_compare.add_trace(go.Scatterpolar(
                r=all_vals + [all_vals[0]], theta=labels + [labels[0]],
                fill="toself", fillcolor="rgba(255,255,255,0.04)",
                line=dict(color="#555", width=1.5, dash="dot"), name="All Valid",
            ))
            fig_compare.update_layout(
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True, range=[0, 1],
                        tickfont=dict(size=8, color="#555"),
                        gridcolor="rgba(255,255,255,0.05)"),
                    angularaxis=dict(tickfont=dict(size=9, color="#8888aa"),
                        gridcolor="rgba(255,255,255,0.05)"),
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(font=dict(color="#aaa"), x=0.85, y=1.1),
                height=380,
                margin=dict(l=60, r=60, t=30, b=30),
            )
            st.plotly_chart(fig_compare, use_container_width=True)

        # Experience distribution
        st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            st.markdown("#### Experience Distribution (Top 15)")
            top15_exp = [(p.candidate_id[-4:], p.experience_years) for _, p, _, _ in scored[:15]]
            if top15_exp:
                fig_exp = go.Figure(go.Bar(
                    x=[e[0] for e in top15_exp],
                    y=[e[1] for e in top15_exp],
                    marker=dict(
                        color=[e[1] for e in top15_exp],
                        colorscale=[[0, "#b0b0b5"], [1, "#e8c547"]],
                    ),
                ))
                fig_exp.update_layout(
                    xaxis=dict(title="Candidate", color="#888", gridcolor="rgba(255,255,255,0.04)"),
                    yaxis=dict(title="Years", color="#888", gridcolor="rgba(255,255,255,0.04)"),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=280,
                    margin=dict(l=40, r=20, t=20, b=40),
                )
                st.plotly_chart(fig_exp, use_container_width=True)

        with exp_col2:
            st.markdown("#### Behavioral Multiplier (Top 15)")
            top15_bm = [(p.candidate_id[-4:], s.behavioral_multiplier) for _, p, s, _ in scored[:15]]
            if top15_bm:
                colors = ["#43e97b" if b >= 1.0 else "#ff6b6b" for _, b in top15_bm]
                fig_bm = go.Figure(go.Bar(
                    x=[b[0] for b in top15_bm],
                    y=[b[1] for b in top15_bm],
                    marker=dict(color=colors),
                ))
                fig_bm.add_hline(y=1.0, line_dash="dash", line_color="#888", annotation_text="Baseline")
                fig_bm.update_layout(
                    xaxis=dict(title="Candidate", color="#888", gridcolor="rgba(255,255,255,0.04)"),
                    yaxis=dict(title="Multiplier", color="#888", gridcolor="rgba(255,255,255,0.04)", range=[0.5, 1.3]),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=280,
                    margin=dict(l=40, r=20, t=20, b=40),
                )
                st.plotly_chart(fig_bm, use_container_width=True)

    # ════════════════════════════════════════════════════════════════
    # TAB 3: HONEYPOT FORENSICS
    # ════════════════════════════════════════════════════════════════
    with tab_honeypots:
        st.markdown("### 🍯 Honeypot Forensics")

        if not honeypots_detected:
            st.success("✅ No honeypots detected in this candidate set. The sample appears clean.")
        else:
            st.markdown(
                f'<span style="color:#a0a0c0;font-size:0.85rem;">'
                f'{len(honeypots_detected)} candidates identified with impossible profile contradictions '
                f'and excluded from ranking.</span>',
                unsafe_allow_html=True,
            )

            # Flag type distribution
            from collections import Counter
            flag_types = Counter()
            for _, hp_result, _ in honeypots_detected:
                for flag in hp_result.flags:
                    if "expert" in flag and "0 months" in flag:
                        flag_types["Expert Skills (0 months)"] += 1
                    elif "dates span" in flag or "since start" in flag:
                        flag_types["Impossible Duration"] += 1
                    elif "company was founded" in flag:
                        flag_types["Pre-Founding Employment"] += 1
                    elif "overlap" in flag:
                        flag_types["Overlapping Roles"] += 1
                    elif "future" in flag:
                        flag_types["Future Dates"] += 1
                    elif "Career total" in flag:
                        flag_types["Career Inflation"] += 1
                    else:
                        flag_types["Other"] += 1

            if flag_types:
                fig_flags = go.Figure(go.Bar(
                    x=list(flag_types.values()),
                    y=list(flag_types.keys()),
                    orientation='h',
                    marker=dict(
                        color=list(flag_types.values()),
                        colorscale=[[0, "rgba(255,80,80,0.4)"], [1, "rgba(255,50,50,0.8)"]],
                    ),
                ))
                fig_flags.update_layout(
                    title=dict(text="Honeypot Flag Distribution", font=dict(color="#ccc", size=13)),
                    xaxis=dict(title="Count", color="#888", gridcolor="rgba(255,255,255,0.04)"),
                    yaxis=dict(color="#aaa"),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=250,
                    margin=dict(l=180, r=30, t=40, b=30),
                )
                st.plotly_chart(fig_flags, use_container_width=True)

            st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

            # Individual honeypot cards
            for profile, hp_result, scores in honeypots_detected:
                title = profile.job_titles[0] if profile.job_titles else "Unknown"
                company = profile.companies[0] if profile.companies else "Unknown"
                flags_html = "".join(
                    f'<span class="tag tag-hp">🚩 {flag}</span>' for flag in hp_result.flags
                )
                st.markdown(f"""
                <div class="hp-card">
                    <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px;">
                        <span class="rank-circle" style="background:rgba(255,50,50,0.2);color:#ff5555;font-size:1.2rem;">🍯</span>
                        <div style="flex:1;">
                            <div style="color:#ff9999;font-weight:700;font-size:0.95rem;">{title} at {company}</div>
                            <div style="color:#886666;font-size:0.75rem;">
                                {profile.candidate_id} · Severity: {hp_result.severity} · {profile.experience_years:.1f} yrs claimed
                            </div>
                        </div>
                    </div>
                    <div>{flags_html}</div>
                </div>
                """, unsafe_allow_html=True)

    # ════════════════════════════════════════════════════════════════
    # TAB 4: METHODOLOGY
    # ════════════════════════════════════════════════════════════════
    with tab_methodology:
        st.markdown("### 🔬 System Architecture & Methodology")

        # Pipeline steps
        st.markdown("#### Pipeline")
        p1, p2, p3, p4, p5, p6 = st.columns(6)
        steps = [
            (p1, "1", "Parse", "Profile normalization"),
            (p2, "2", "Embed", "sentence-transformers"),
            (p3, "3", "Score", "12 weighted signals"),
            (p4, "4", "Filter", "Honeypot detection"),
            (p5, "5", "Explain", "Per-candidate reasoning"),
            (p6, "6", "Export", "submission.csv"),
        ]
        for col, num, label, desc in steps:
            with col:
                st.markdown(f"""
                <div class="pipe-step">
                    <div class="pipe-num">{num}</div><br>
                    <span class="pipe-label">{label}</span><br>
                    <span class="pipe-desc">{desc}</span>
                </div>
                """, unsafe_allow_html=True)

        st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

        mc1, mc2 = st.columns(2)
        with mc1:
            st.markdown("""
            <div class="info-panel">
                <h4>🧠 Evidence-First Embeddings</h4>
                <p>
                Career descriptions are prioritized over self-declared skill lists to defeat
                keyword stuffing. We encode <b>what candidates actually did</b> (shipped
                production systems, ran A/B tests, built retrieval pipelines) rather than
                what they claim to know. This is achieved using <b>all-MiniLM-L6-v2</b>
                sentence-transformers with cosine similarity against a rich JD embedding.
                </p>
            </div>
            """, unsafe_allow_html=True)

        with mc2:
            st.markdown("""
            <div class="info-panel">
                <h4>📊 12-Signal Composite Scoring</h4>
                <p>
                Each candidate is scored across 12 weighted signals: <b>career evidence</b>
                (production retrieval/ranking keywords), <b>skill validation</b> (duration,
                proficiency, endorsements), <b>career stability</b> (penalizes title-chasing),
                <b>product-vs-services fit</b>, location, culture, and work-mode compatibility.
                A bounded <b>behavioral multiplier</b> (0.65×–1.25×) uses Redrob signals.
                </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")
        mc3, mc4 = st.columns(2)
        with mc3:
            st.markdown("""
            <div class="info-panel">
                <h4>🍯 Honeypot Detection (7 Flags)</h4>
                <p>
                A severity-threshold detector catches impossible profiles: expert skills with
                0 months usage, role durations exceeding calendar time, future dates, overlapping
                full-time roles, career sum inflation, and <b>pre-founding employment</b> at
                recently-founded startups. 113 honeypots caught across 100K candidates with
                <b>0% false positive rate</b> on clean validation samples.
                </p>
            </div>
            """, unsafe_allow_html=True)

        with mc4:
            st.markdown("""
            <div class="info-panel">
                <h4>💬 Dynamic Explainability</h4>
                <p>
                Every ranked candidate receives a profile-specific, plain-language reasoning
                paragraph that references their <b>actual job title, company, matched skills,
                and behavioral traits</b>. No templates — each explanation is dynamically
                generated from the candidate's real signal scores and profile data, ensuring
                full transparency and auditability for Stage 4 review.
                </p>
            </div>
            """, unsafe_allow_html=True)

    # ── Download ──────────────────────────────────────────────────────
    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=["candidate_id", "rank", "score", "reasoning"])
    writer.writeheader()
    max_score_dl = scored[0][0] if scored else 1.0
    for rank_idx, (score, profile, scores, reasoning) in enumerate(scored[:100], start=1):
        writer.writerow({
            "candidate_id": profile.candidate_id,
            "rank": rank_idx,
            "score": f"{score:.6f}",
            "reasoning": reasoning.replace("\n", " ").strip(),
        })
    st.download_button(
        label="📥 Download submission.csv",
        data=csv_buffer.getvalue(),
        file_name="submission.csv",
        mime="text/csv",
        use_container_width=True,
    )


# ── LANDING STATE (no results yet) ───────────────────────────────────────
elif "results" not in st.session_state:
    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)

    st.info(
        "👈 **Upload a JSONL or JSON candidate file** in the sidebar and click "
        "**🚀 Run Pipeline** to begin ranking.",
        icon="💡",
    )

    # Feature cards
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        st.markdown("""
        <div class="glass-card" style="text-align:center;">
            <div style="font-size:2rem;margin-bottom:8px;">🧠</div>
            <div class="glass-value" style="font-size:1.4rem;">AI</div>
            <div class="glass-label">Evidence-First Embeddings</div>
        </div>""", unsafe_allow_html=True)
    with f2:
        st.markdown("""
        <div class="glass-card" style="text-align:center;">
            <div style="font-size:2rem;margin-bottom:8px;">📊</div>
            <div class="glass-value" style="font-size:1.4rem;">12</div>
            <div class="glass-label">Weighted Signals</div>
        </div>""", unsafe_allow_html=True)
    with f3:
        st.markdown("""
        <div class="glass-card" style="text-align:center;">
            <div style="font-size:2rem;margin-bottom:8px;">🍯</div>
            <div class="glass-value" style="font-size:1.4rem;">113</div>
            <div class="glass-label">Honeypots Detected</div>
        </div>""", unsafe_allow_html=True)
    with f4:
        st.markdown("""
        <div class="glass-card" style="text-align:center;">
            <div style="font-size:2rem;margin-bottom:8px;">🇮🇳</div>
            <div class="glass-value" style="font-size:1.4rem;">100K</div>
            <div class="glass-label">India-Scale Pipeline</div>
        </div>""", unsafe_allow_html=True)

    # Architecture pipeline
    st.markdown('<div class="glow-divider"></div>', unsafe_allow_html=True)
    st.markdown("#### 🏗️ Pipeline Architecture")
    a1, a2, a3, a4, a5, a6 = st.columns(6)
    arch_steps = [
        (a1, "1", "Upload", "JSONL / JSON"),
        (a2, "2", "Embed", "MiniLM-L6-v2"),
        (a3, "3", "Score", "12 Signals"),
        (a4, "4", "Filter", "7 HP Flags"),
        (a5, "5", "Explain", "Dynamic NLG"),
        (a6, "6", "Export", "CSV Download"),
    ]
    for col, num, label, desc in arch_steps:
        with col:
            st.markdown(f"""
            <div class="pipe-step">
                <div class="pipe-num">{num}</div><br>
                <span class="pipe-label">{label}</span><br>
                <span class="pipe-desc">{desc}</span>
            </div>
            """, unsafe_allow_html=True)
