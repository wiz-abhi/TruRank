"""
Streamlit demo — Intelligent Candidate Ranking System
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

# Ensure project root on import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import plotly.graph_objects as go
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

# ── Custom CSS ───────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 40%, #24243e 100%);
    }

    /* Hero */
    .hero-title {
        font-size: 2.6rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .hero-subtitle {
        color: #a0a0c0;
        font-size: 1.05rem;
        margin-top: -0.5rem;
    }

    /* Metric cards */
    .metric-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 1.2rem 1.4rem;
        backdrop-filter: blur(12px);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.15);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #fff;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8888aa;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Candidate card */
    .candidate-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 14px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
        transition: all 0.2s ease;
    }
    .candidate-card:hover {
        background: rgba(255,255,255,0.07);
        border-color: rgba(102, 126, 234, 0.3);
    }

    /* Honeypot card (red tint) */
    .honeypot-card {
        background: rgba(255,50,50,0.06);
        border: 1px solid rgba(255,80,80,0.25);
        border-radius: 14px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
    }

    /* Rank badges */
    .rank-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 36px;
        height: 36px;
        border-radius: 50%;
        font-weight: 700;
        font-size: 0.95rem;
        margin-right: 0.8rem;
    }
    .rank-1 { background: linear-gradient(135deg, #FFD700, #FFA500); color: #1a1a2e; }
    .rank-2 { background: linear-gradient(135deg, #C0C0C0, #A8A8A8); color: #1a1a2e; }
    .rank-3 { background: linear-gradient(135deg, #CD7F32, #B87333); color: #1a1a2e; }
    .rank-n { background: rgba(102, 126, 234, 0.2); color: #667eea; }

    /* Progress bar */
    .match-bar-bg {
        background: rgba(255,255,255,0.1);
        border-radius: 6px;
        height: 8px;
        width: 100%;
        overflow: hidden;
    }
    .match-bar-fill {
        height: 100%;
        border-radius: 6px;
        background: linear-gradient(90deg, #667eea, #764ba2);
        transition: width 0.6s ease;
    }

    /* Reason tag */
    .reason-tag {
        display: inline-block;
        background: rgba(102, 126, 234, 0.15);
        color: #a0b0ff;
        font-size: 0.78rem;
        padding: 3px 10px;
        border-radius: 20px;
        margin-right: 6px;
        margin-top: 4px;
    }
    .warning-tag {
        display: inline-block;
        background: rgba(255, 107, 107, 0.15);
        color: #ff6b6b;
        font-size: 0.78rem;
        padding: 3px 10px;
        border-radius: 20px;
        margin-right: 6px;
        margin-top: 4px;
    }
    .honeypot-tag {
        display: inline-block;
        background: rgba(255, 50, 50, 0.2);
        color: #ff4444;
        font-size: 0.78rem;
        padding: 3px 10px;
        border-radius: 20px;
        margin-right: 6px;
        margin-top: 4px;
        font-weight: 600;
    }

    /* Signal bars */
    .signal-row {
        display: flex;
        align-items: center;
        margin-bottom: 4px;
    }
    .signal-label {
        width: 150px;
        font-size: 0.75rem;
        color: #8888aa;
    }
    .signal-bar-bg {
        flex: 1;
        background: rgba(255,255,255,0.08);
        border-radius: 4px;
        height: 6px;
        overflow: hidden;
    }
    .signal-bar-fill {
        height: 100%;
        border-radius: 4px;
    }
    .signal-val {
        width: 40px;
        text-align: right;
        font-size: 0.75rem;
        color: #ccc;
        margin-left: 8px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(15,12,41,0.95) !important;
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    /* Info sections */
    .how-section {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 1.5rem;
        margin-top: 2rem;
    }
    .how-section h3 { color: #667eea; }
    .how-section p { color: #a0a0c0; font-size: 0.9rem; line-height: 1.6; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Helper functions ─────────────────────────────────────────────────────
def _rank_badge(rank: int) -> str:
    medals = {1: "🥇", 2: "🥈", 3: "🥉"}
    cls = f"rank-{rank}" if rank <= 3 else "rank-n"
    label = medals.get(rank, str(rank))
    return f'<span class="rank-badge {cls}">{label}</span>'


def _match_bar(pct: int) -> str:
    return f"""
    <div style="display:flex;align-items:center;gap:8px;">
        <div class="match-bar-bg"><div class="match-bar-fill" style="width:{pct}%"></div></div>
        <span style="color:#fff;font-weight:600;font-size:0.9rem;">{pct}%</span>
    </div>
    """


def _signal_bar(label: str, value: float, color: str = "#667eea") -> str:
    pct = int(max(0, min(100, value * 100)))
    return f"""
    <div class="signal-row">
        <span class="signal-label">{label}</span>
        <div class="signal-bar-bg">
            <div class="signal-bar-fill" style="width:{pct}%;background:{color};"></div>
        </div>
        <span class="signal-val">{value:.2f}</span>
    </div>
    """


# Signal display configuration — matches current architecture
SIGNAL_CONFIG = {
    "career_evidence":    {"label": "Career Evidence",    "color": "#43e97b"},
    "semantic_similarity":{"label": "Semantic Match",     "color": "#667eea"},
    "skill_evidence":     {"label": "Skill Evidence",     "color": "#764ba2"},
    "skill_match":        {"label": "Skill Match",        "color": "#f093fb"},
    "experience_fit":     {"label": "Experience Fit",     "color": "#38f9d7"},
    "career_stability":   {"label": "Career Stability",   "color": "#fee140"},
    "product_company_fit":{"label": "Product Co. Fit",    "color": "#fa709a"},
    "domain_alignment":   {"label": "Domain Alignment",   "color": "#30cfd0"},
    "location_fit":       {"label": "Location Fit",       "color": "#a18cd1"},
    "culture_fit":        {"label": "Culture Fit",        "color": "#ffecd2"},
    "skill_recency":      {"label": "Skill Depth",        "color": "#fcb69f"},
    "work_mode_fit":      {"label": "Work Mode Fit",      "color": "#84fab0"},
    "profile_trust":      {"label": "Profile Trust",      "color": "#ff6b6b"},
}


# ── Cached model loading ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# ── SIDEBAR ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🇮🇳 IndiaRanks")
    st.markdown(
        '<span style="color:#8888aa;font-size:0.8rem;">Evidence-First Candidate Ranking</span>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown("**📂 Candidate Data**")
    uploaded = st.file_uploader(
        "Upload candidates (JSONL or JSON)",
        type=["jsonl", "json"],
        key="candidates_upload",
    )

    st.markdown("---")
    num_show = st.slider("Top candidates to show", 5, 100, 15, key="num_show")
    show_signals = st.toggle("Show signal breakdown", value=True, key="show_signals")
    show_honeypots = st.toggle("Show detected honeypots", value=True, key="show_honeypots")

    st.markdown("---")
    run_btn = st.button("🚀 Rank Candidates", use_container_width=True, type="primary")


# ── MAIN PANEL ───────────────────────────────────────────────────────────
st.markdown('<h1 class="hero-title">IndiaRanks</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-subtitle">Evidence-First Candidate Ranking · Semantic AI + Honeypot Detection + Behavioral Signals</p>',
    unsafe_allow_html=True,
)

if run_btn and uploaded is not None:
    # Parse uploaded candidates
    raw_candidates = []
    content = uploaded.read().decode("utf-8")
    if uploaded.name.endswith(".jsonl"):
        for line in content.strip().split("\n"):
            if line.strip():
                raw_candidates.append(json.loads(line))
    else:
        data = json.loads(content)
        raw_candidates = data if isinstance(data, list) else [data]

    if not raw_candidates:
        st.error("No candidates found in the uploaded file.")
        st.stop()

    with st.spinner("⚡ Running full ranking pipeline…"):
        # 1. Setup
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

        # 2. Parse profiles
        parser = ProfileParser()
        profiles = parser.parse_many(raw_candidates)

        # 3. Compute embeddings
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

        # 4. Semantic similarity
        jd_norm = np.asarray(jd_emb, dtype=np.float32)
        prof_norms = np.asarray(profile_embs, dtype=np.float32)
        semantic_scores = prof_norms @ jd_norm

        # 5. Compute signals
        computer = SignalComputer()
        explainer = ExplainerEngine()
        detector = HoneypotDetector()

        scored = []
        honeypots_detected = []

        for i, profile in enumerate(profiles):
            sim = float(semantic_scores[i])
            scores = computer.compute_all(profile, jd, sim)
            hp_result = detector.detect(profile.raw)

            if hp_result.is_honeypot:
                honeypots_detected.append((profile, hp_result, scores))
            else:
                reasoning = explainer.explain_rank(profile, scores, jd)
                scored.append((scores.composite_score, profile, scores, reasoning))

        # Sort by score descending
        scored.sort(key=lambda x: (-x[0], x[1].candidate_id))
        top_n = scored[:num_show]

    # ── Summary Metrics ──────────────────────────────────────────────
    st.markdown("---")
    cols = st.columns(5)
    with cols[0]:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{len(profiles)}</div>'
            f'<div class="metric-label">Total Candidates</div></div>',
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{len(honeypots_detected)}</div>'
            f'<div class="metric-label">Honeypots Caught</div></div>',
            unsafe_allow_html=True,
        )
    with cols[2]:
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{len(scored)}</div>'
            f'<div class="metric-label">Valid Candidates</div></div>',
            unsafe_allow_html=True,
        )
    with cols[3]:
        top_score = top_n[0][0] if top_n else 0
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{top_score:.4f}</div>'
            f'<div class="metric-label">Top Score</div></div>',
            unsafe_allow_html=True,
        )
    with cols[4]:
        hp_pct = (len(honeypots_detected) / max(len(profiles), 1)) * 100
        st.markdown(
            f'<div class="metric-card"><div class="metric-value">{hp_pct:.1f}%</div>'
            f'<div class="metric-label">Honeypot Rate</div></div>',
            unsafe_allow_html=True,
        )

    # ── JD Summary ───────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📋 Target Job Description")
    jd_cols = st.columns(3)
    with jd_cols[0]:
        st.markdown("**Required Skills**")
        st.markdown(" · ".join([f"`{s}`" for s in jd.required_skills[:12]]))
    with jd_cols[1]:
        st.markdown("**Preferred Skills**")
        st.markdown(" · ".join([f"`{s}`" for s in jd.preferred_skills]))
    with jd_cols[2]:
        st.markdown(f"**Experience:** {jd.min_experience_years:.0f}+ years")
        st.markdown(f"**Domain:** {jd.domain.replace('_', ' ').title()}")
        st.markdown(f"**Culture:** {', '.join(jd.culture_signals)}")

    # ── Ranked Candidates ────────────────────────────────────────────
    st.markdown("---")
    st.markdown(f"### 🏆 Top {len(top_n)} Ranked Candidates")

    max_score = top_n[0][0] if top_n else 1.0

    for rank_idx, (score, profile, scores, reasoning) in enumerate(top_n, start=1):
        badge = _rank_badge(rank_idx)
        norm_pct = int((score / max_score) * 100) if max_score > 0 else 0
        bar = _match_bar(norm_pct)

        title = profile.job_titles[0] if profile.job_titles else "Professional"
        company = profile.companies[0] if profile.companies else ""
        loc = profile.location or ""

        # Build reason tags from reasoning text
        reasoning_clean = reasoning.replace("\n", " ").strip()

        # Determine warning tags
        warning_html = ""
        if scores.profile_trust < 0.7:
            warning_html += '<span class="warning-tag">⚠ Low profile trust</span>'
        if scores.career_stability < 0.4:
            warning_html += '<span class="warning-tag">⚠ Unstable career</span>'
        if scores.behavioral_multiplier < 0.8:
            warning_html += '<span class="warning-tag">⚠ Low activity</span>'

        loc_html = f' · 📍 {loc}' if loc else ''

        st.markdown(
            f"""
            <div class="candidate-card">
                <div style="display:flex;align-items:center;margin-bottom:8px;">
                    {badge}
                    <div style="flex:1;">
                        <div style="color:#fff;font-weight:600;font-size:1.05rem;">{title} at {company}</div>
                        <div style="color:#8888aa;font-size:0.8rem;">
                            {profile.candidate_id} · {profile.experience_years:.1f} yrs exp{loc_html}
                        </div>
                    </div>
                    <div style="width:180px;">{bar}</div>
                </div>
                <div style="color:#c0c0d8;font-size:0.82rem;margin-bottom:6px;">{reasoning_clean}</div>
                {warning_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

        if show_signals:
            with st.expander(f"📊 Signal Breakdown — {profile.candidate_id}", expanded=False):
                scores_dict = scores.to_dict()

                # Signal bars
                signal_html = ""
                for key, cfg_item in SIGNAL_CONFIG.items():
                    val = scores_dict.get(key, 0)
                    signal_html += _signal_bar(cfg_item["label"], val, cfg_item["color"])

                # Behavioral multiplier (displayed separately)
                bm = scores_dict.get("behavioral_multiplier", 1.0)
                bm_color = "#43e97b" if bm >= 1.0 else "#ff6b6b"
                signal_html += f"""
                <div style="margin-top:8px;padding-top:6px;border-top:1px solid rgba(255,255,255,0.1);">
                    <div class="signal-row">
                        <span class="signal-label" style="font-weight:600;">Behavioral ×</span>
                        <div class="signal-bar-bg">
                            <div class="signal-bar-fill" style="width:{int(bm/1.25*100)}%;background:{bm_color};"></div>
                        </div>
                        <span class="signal-val" style="font-weight:600;">{bm:.2f}×</span>
                    </div>
                </div>
                """
                st.markdown(signal_html, unsafe_allow_html=True)

                # Radar chart
                labels = [v["label"] for v in SIGNAL_CONFIG.values()]
                values = [scores_dict.get(k, 0) for k in SIGNAL_CONFIG.keys()]
                values_closed = values + [values[0]]
                labels_closed = labels + [labels[0]]

                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(
                    r=values_closed,
                    theta=labels_closed,
                    fill="toself",
                    fillcolor="rgba(102, 126, 234, 0.15)",
                    line=dict(color="#667eea", width=2),
                    marker=dict(size=5, color="#764ba2"),
                ))
                fig.update_layout(
                    polar=dict(
                        bgcolor="rgba(0,0,0,0)",
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1],
                            tickfont=dict(size=9, color="#666"),
                            gridcolor="rgba(255,255,255,0.08)",
                        ),
                        angularaxis=dict(
                            tickfont=dict(size=10, color="#a0a0c0"),
                            gridcolor="rgba(255,255,255,0.08)",
                        ),
                    ),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=60, r=60, t=20, b=20),
                    height=320,
                    showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)

    # ── Honeypot Analysis Panel ──────────────────────────────────────
    if show_honeypots and honeypots_detected:
        st.markdown("---")
        st.markdown(f"### 🍯 Honeypots Detected ({len(honeypots_detected)})")
        st.markdown(
            '<span style="color:#a0a0c0;font-size:0.85rem;">'
            "These candidates have impossible profile contradictions and were excluded from the ranking."
            "</span>",
            unsafe_allow_html=True,
        )

        for profile, hp_result, scores in honeypots_detected:
            title = profile.job_titles[0] if profile.job_titles else "Unknown"
            company = profile.companies[0] if profile.companies else "Unknown"
            flags_html = "".join(
                f'<span class="honeypot-tag">🚩 {flag}</span>' for flag in hp_result.flags
            )

            st.markdown(
                f"""
                <div class="honeypot-card">
                    <div style="display:flex;align-items:center;margin-bottom:8px;">
                        <span class="rank-badge" style="background:rgba(255,50,50,0.3);color:#ff4444;">🍯</span>
                        <div style="flex:1;">
                            <div style="color:#ff8888;font-weight:600;font-size:1rem;">{title} at {company}</div>
                            <div style="color:#aa6666;font-size:0.8rem;">
                                {profile.candidate_id} · Severity: {hp_result.severity} · {profile.experience_years:.1f} yrs claimed
                            </div>
                        </div>
                    </div>
                    <div>{flags_html}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # ── Download submission CSV ──────────────────────────────────────
    st.markdown("---")
    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=["candidate_id", "rank", "score", "reasoning"])
    writer.writeheader()
    for rank_idx, (score, profile, scores, reasoning) in enumerate(scored[:100], start=1):
        writer.writerow({
            "candidate_id": profile.candidate_id,
            "rank": rank_idx,
            "score": f"{score / max_score:.6f}" if max_score > 0 else "0.000000",
            "reasoning": reasoning.replace("\n", " ").strip(),
        })
    st.download_button(
        label="📥 Download submission.csv",
        data=csv_buffer.getvalue(),
        file_name="submission.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # ── How This Works ───────────────────────────────────────────────
    st.markdown(
        """
        <div class="how-section">
            <h3>🔬 How This Works</h3>
            <p>
            This system uses <b>evidence-first semantic embeddings</b> — career descriptions
            are prioritized over self-declared skill lists to defeat keyword stuffing.
            Each candidate is scored across <b>12 weighted signals</b> including career evidence
            (production retrieval/ranking systems), skill claim validation (duration, proficiency,
            assessments), career stability (penalizing title-chasing), and product-vs-services fit.
            </p>
            <p>
            A <b>bounded behavioral multiplier</b> (0.65×–1.25×) uses Redrob signals like
            notice period, GitHub activity, response rate, and profile completeness to re-rank
            technically similar candidates without manufacturing false relevance.
            </p>
            <p>
            The <b>Honeypot Detector</b> catches ~80 impossible profiles using 7 hard/medium
            flags: expert skills with 0 months usage, career durations exceeding calendar time,
            future role dates, overlapping full-time roles, and graduation-timeline violations.
            Submissions with >10% honeypot rate in the top 100 are auto-disqualified.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

elif not run_btn:
    # Landing state
    st.markdown("---")
    st.info(
        "👈 **Upload a JSONL/JSON candidate file** in the sidebar and click "
        "**🚀 Rank Candidates** to begin.",
        icon="💡",
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(
            '<div class="metric-card"><div class="metric-value">12</div>'
            '<div class="metric-label">Weighted Signals</div></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            '<div class="metric-card"><div class="metric-value">🍯</div>'
            '<div class="metric-label">Honeypot Detection</div></div>',
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            '<div class="metric-card"><div class="metric-value">🧠</div>'
            '<div class="metric-label">Evidence-First AI</div></div>',
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            '<div class="metric-card"><div class="metric-value">🇮🇳</div>'
            '<div class="metric-label">India-Native</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        """
        <div class="how-section" style="margin-top:1.5rem;">
            <h3>🏗️ Architecture</h3>
            <p>
            <b>Step 1 — Upload:</b> Provide a JSONL or JSON file of candidate profiles.<br>
            <b>Step 2 — Embed:</b> Profiles are encoded with sentence-transformers (all-MiniLM-L6-v2).<br>
            <b>Step 3 — Score:</b> 12 weighted signals compute a composite score per candidate.<br>
            <b>Step 4 — Filter:</b> Honeypot detector removes impossible profiles.<br>
            <b>Step 5 — Explain:</b> Each candidate receives specific, plain-language reasoning.<br>
            <b>Step 6 — Download:</b> Export submission.csv ready for the hackathon portal.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
