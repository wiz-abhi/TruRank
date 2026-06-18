import streamlit as st
import pandas as pd
import ast

st.set_page_config(
    page_title="IndiaRanks Results Dashboard",
    page_icon="🇮🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for the WOW factor
st.markdown("""
<style>
    .metric-card {
        background-color: #1E1E2E;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
        border-left: 5px solid #6366F1;
        transition: transform 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(99, 102, 241, 0.4);
    }
    .candidate-id {
        font-size: 24px;
        font-weight: 800;
        color: #F8FAFC;
        margin-bottom: 5px;
    }
    .rank-score {
        font-size: 16px;
        color: #94A3B8;
        margin-bottom: 15px;
    }
    .score-badge {
        background-color: #4F46E5;
        color: white;
        padding: 4px 10px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 14px;
    }
    .reasoning-list {
        color: #CBD5E1;
        font-size: 14px;
        line-height: 1.6;
        padding-left: 20px;
    }
    .header-title {
        font-size: 3rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #4F46E5, #EC4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header-title">IndiaRanks Candidate Dashboard</div>', unsafe_allow_html=True)
st.markdown("### 🏆 Top 100 Ranked Candidates for Senior AI/ML Engineer")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("submission.csv")
        return df
    except FileNotFoundError:
        return None

df = load_data()

if df is None:
    st.error("submission.csv not found! Please run `python rank.py` first.")
    st.stop()

# Sidebar
st.sidebar.markdown("## ⚙️ Filter Candidates")
top_n = st.sidebar.slider("Show Top N Candidates", min_value=5, max_value=100, value=20)
min_score = st.sidebar.slider("Minimum Composite Score", min_value=0.0, max_value=1.0, value=0.0, step=0.05)

filtered_df = df[(df['rank'] <= top_n) & (df['score'] >= min_score)]

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 About IndiaRanks")
st.sidebar.markdown(
    "This dashboard visualizes the results of our **IndiaRanks** engine. "
    "Candidates are scored using a two-step pipeline that leverages **semantic embeddings** "
    "and **9 distinct behavioral signals** tailored for the Indian talent market."
)

st.markdown(f"Showing **{len(filtered_df)}** candidates.")

# Render candidates
for _, row in filtered_df.iterrows():
    # Attempt to parse reasoning if it's a string list, else treat as single string
    reasoning_raw = row['reasoning']
    try:
        reasons = ast.literal_eval(reasoning_raw)
        if not isinstance(reasons, list):
            reasons = [reasoning_raw]
    except (ValueError, SyntaxError):
        reasons = [r.strip() for r in reasoning_raw.split('\n') if r.strip()]

    reasons_html = "".join([f"<li>{r}</li>" for r in reasons])
    
    st.markdown(f"""
        <div class="metric-card">
            <div class="candidate-id">{row['candidate_id']}</div>
            <div class="rank-score">
                Rank: <b>#{row['rank']}</b> &nbsp;&nbsp;|&nbsp;&nbsp; 
                <span class="score-badge">Score: {row['score']:.4f}</span>
            </div>
            <div style="font-weight: 600; color: #F1F5F9; margin-bottom: 8px;">Why this candidate?</div>
            <ul class="reasoning-list">
                {reasons_html}
            </ul>
        </div>
    """, unsafe_allow_html=True)
