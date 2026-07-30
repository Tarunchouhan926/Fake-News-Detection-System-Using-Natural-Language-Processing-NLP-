"""
Fake News Detection using NLP — Premium SaaS UI
--------------------------------------------------
Frontend has been completely redesigned for a modern, premium AI-product feel.
The ML logic (clean_text, predict_news, model/vectorizer loading) is UNCHANGED
from the original app, exactly as required.
"""

import re
import time
import pickle
import streamlit as st

# ============================================================
# PAGE CONFIG (must be first Streamlit call)
# ============================================================
st.set_page_config(
    page_title="Fake News Detection | AI-Powered NLP",
    page_icon="🛰️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ============================================================
# ML BACKEND — UNTOUCHED (as required by the brief)
# ============================================================
@st.cache_resource
def load_artifacts():
    with open("logistic_regression_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)
    return model, tfidf


model, tfidf = load_artifacts()


# ---- Same cleaning function used during training ----
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def predict_news(news_text):
    cleaned = clean_text(news_text)
    vectorized = tfidf.transform([cleaned])
    prob = model.predict_proba(vectorized)[0][1]
    label = "True" if prob > 0.5 else "Fake"
    return label, float(prob)


# ============================================================
# SESSION STATE
# ============================================================
if "news_text" not in st.session_state:
    st.session_state.news_text = ""
if "result" not in st.session_state:
    st.session_state.result = None  # (label, prob)
if "has_run" not in st.session_state:
    st.session_state.has_run = False


# ============================================================
# GLOBAL CSS — theme, typography, components, animations
# ============================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500&display=swap');

    :root{
        --navy:#0F172A;
        --blue:#2563EB;
        --cyan:#38BDF8;
        --green:#10B981;
        --red:#EF4444;
        --bg:#F8FAFC;
        --card:#FFFFFF;
        --border:#E2E8F0;
        --muted:#64748B;
    }

    html, body, [class*="css"]{
        font-family:'Inter', sans-serif;
    }

    /* Hide default Streamlit chrome */
    #MainMenu, header, footer {visibility:hidden;}
    div[data-testid="stToolbar"]{visibility:hidden;}
    div[data-testid="stDecoration"]{display:none;}
    .block-container{
        padding-top:2.2rem;
        padding-bottom:3rem;
        max-width:820px;
    }

    body{ background:var(--bg); }
    .stApp{
        background:
            radial-gradient(circle at 10% 0%, rgba(56,189,248,0.10) 0%, rgba(56,189,248,0) 40%),
            radial-gradient(circle at 90% 10%, rgba(37,99,235,0.08) 0%, rgba(37,99,235,0) 45%),
            var(--bg);
    }

    /* Custom scrollbar */
    ::-webkit-scrollbar{ width:8px; height:8px; }
    ::-webkit-scrollbar-track{ background:transparent; }
    ::-webkit-scrollbar-thumb{ background:#CBD5E1; border-radius:10px; }
    ::-webkit-scrollbar-thumb:hover{ background:#94A3B8; }

    /* ---------------- HERO ---------------- */
    .hero-wrap{ text-align:center; padding:1.2rem 0 0.4rem 0; }
    .hero-icon{
        width:74px; height:74px; margin:0 auto 1.1rem auto;
        display:flex; align-items:center; justify-content:center;
        border-radius:22px;
        background:linear-gradient(135deg, var(--navy) 0%, var(--blue) 60%, var(--cyan) 100%);
        box-shadow:0 12px 28px rgba(37,99,235,0.35), 0 4px 10px rgba(15,23,42,0.15);
        font-size:2.1rem;
        animation:float 4s ease-in-out infinite;
    }
    @keyframes float{
        0%,100%{ transform:translateY(0px); }
        50%{ transform:translateY(-8px); }
    }
    .hero-title{
        font-size:2.35rem; font-weight:900; color:var(--navy);
        letter-spacing:-0.02em; line-height:1.15; margin-bottom:0.55rem;
    }
    .hero-title .grad{
        background:linear-gradient(90deg, var(--blue), var(--cyan));
        -webkit-background-clip:text; background-clip:text; color:transparent;
    }
    .hero-sub{
        font-size:1.05rem; color:var(--muted); font-weight:500;
        max-width:520px; margin:0 auto 1.1rem auto;
    }
    .badge-row{ display:flex; justify-content:center; gap:0.6rem; flex-wrap:wrap; margin-bottom:1.6rem; }
    .badge{
        display:inline-flex; align-items:center; gap:0.4rem;
        background:#ECFDF5; color:#047857; border:1px solid #A7F3D0;
        padding:0.35rem 0.9rem; border-radius:999px;
        font-size:0.82rem; font-weight:700;
    }
    .badge.blue{
        background:#EFF6FF; color:#1D4ED8; border:1px solid #BFDBFE;
    }
    .divider{
        height:1px; width:100%;
        background:linear-gradient(90deg, transparent, var(--border) 20%, var(--border) 80%, transparent);
        margin:0.4rem 0 1.8rem 0;
    }

    /* ---------------- CARD SHELL ---------------- */
    .app-card{
        background:var(--card);
        border:1px solid var(--border);
        border-radius:20px;
        padding:1.9rem 2rem 2rem 2rem;
        box-shadow:0 1px 2px rgba(15,23,42,0.04), 0 12px 32px -12px rgba(15,23,42,0.10);
        margin-bottom:1.6rem;
    }
    .card-header{
        display:flex; align-items:center; gap:0.6rem; margin-bottom:1.1rem;
    }
    .card-header .icon{
        width:38px; height:38px; border-radius:11px;
        background:linear-gradient(135deg, var(--blue), var(--cyan));
        display:flex; align-items:center; justify-content:center;
        font-size:1.1rem; box-shadow:0 6px 14px rgba(37,99,235,0.28);
    }
    .card-header .title{
        font-size:1.15rem; font-weight:800; color:var(--navy);
    }
    .card-header .desc{ font-size:0.85rem; color:var(--muted); margin-top:-2px; }

    /* Textarea styling */
    .stTextArea textarea{
        border-radius:14px !important;
        border:1.5px solid var(--border) !important;
        background:#F8FAFC !important;
        font-size:0.98rem !important;
        color:var(--navy) !important;
        padding:1rem !important;
        transition:all 0.2s ease;
        font-family:'Inter', sans-serif !important;
    }
    .stTextArea textarea:focus{
        border:1.5px solid var(--blue) !important;
        box-shadow:0 0 0 4px rgba(37,99,235,0.12) !important;
        background:#FFFFFF !important;
    }

    .meta-row{
        display:flex; justify-content:space-between; align-items:center;
        margin-top:0.5rem; font-size:0.8rem; color:var(--muted); font-weight:600;
    }
    .meta-row span.tag{
        background:#F1F5F9; padding:0.2rem 0.65rem; border-radius:8px;
        font-family:'JetBrains Mono', monospace; font-size:0.75rem;
    }

    /* Buttons */
    .stButton>button{
        border-radius:12px !important;
        font-weight:700 !important;
        transition:all 0.18s ease !important;
        border:1.5px solid var(--border) !important;
    }
    div[data-testid="column"]:nth-of-type(1) .stButton>button{
        background:#FFFFFF !important; color:var(--navy) !important;
    }
    div[data-testid="column"]:nth-of-type(1) .stButton>button:hover{
        border-color:var(--blue) !important; color:var(--blue) !important;
        transform:translateY(-1px);
    }

    /* Primary Analyze button */
    .stButton>button[kind="primary"]{
        background:linear-gradient(90deg, var(--navy), var(--blue)) !important;
        color:white !important; border:none !important;
        padding:0.75rem 1rem !important; font-size:1.02rem !important;
        box-shadow:0 10px 24px rgba(37,99,235,0.30) !important;
    }
    .stButton>button[kind="primary"]:hover{
        transform:translateY(-2px) scale(1.005);
        box-shadow:0 14px 30px rgba(37,99,235,0.40) !important;
    }
    .stButton>button[kind="primary"]:active{ transform:translateY(0px); }

    /* ---------------- LOADING ---------------- */
    .loading-step{
        display:flex; align-items:center; gap:0.6rem;
        font-size:0.92rem; color:var(--muted); font-weight:600;
        padding:0.35rem 0;
    }
    .loading-step .dot{
        width:8px; height:8px; border-radius:50%;
        background:var(--cyan); animation:pulse 1s infinite ease-in-out;
    }
    @keyframes pulse{
        0%,100%{ opacity:0.3; transform:scale(0.85); }
        50%{ opacity:1; transform:scale(1.1); }
    }

    /* ---------------- RESULT CARD ---------------- */
    .result-shell{
        border-radius:22px; padding:2.2rem; text-align:center;
        border:1.5px solid var(--border);
        animation:fadeUp 0.5s ease;
        margin-bottom:1.6rem;
    }
    @keyframes fadeUp{
        from{ opacity:0; transform:translateY(14px); }
        to{ opacity:1; transform:translateY(0); }
    }
    .result-shell.fake{
        background:linear-gradient(180deg, #FEF2F2 0%, #FFFFFF 65%);
        border-color:#FECACA;
    }
    .result-shell.true{
        background:linear-gradient(180deg, #ECFDF5 0%, #FFFFFF 65%);
        border-color:#A7F3D0;
    }
    .result-icon-wrap{
        width:82px; height:82px; margin:0 auto 1rem auto;
        border-radius:50%; display:flex; align-items:center; justify-content:center;
        font-size:2.6rem;
        animation:pop 0.45s cubic-bezier(.34,1.56,.64,1);
    }
    @keyframes pop{
        0%{ transform:scale(0.4); opacity:0; }
        100%{ transform:scale(1); opacity:1; }
    }
    .result-icon-wrap.fake{ background:#FEE2E2; box-shadow:0 0 0 10px #FEF2F2; }
    .result-icon-wrap.true{ background:#D1FAE5; box-shadow:0 0 0 10px #ECFDF5; }

    .result-label{ font-size:1.7rem; font-weight:900; letter-spacing:-0.01em; margin-bottom:0.2rem; }
    .result-label.fake{ color:var(--red); }
    .result-label.true{ color:var(--green); }
    .result-caption{ color:var(--muted); font-size:0.92rem; font-weight:500; margin-bottom:1.4rem; }

    .conf-num{ font-size:2.4rem; font-weight:900; color:var(--navy); font-family:'JetBrains Mono', monospace; }
    .conf-label{ font-size:0.78rem; color:var(--muted); font-weight:700; letter-spacing:0.06em; text-transform:uppercase; margin-bottom:0.6rem; }

    .meter-track{
        width:100%; height:14px; background:#E2E8F0; border-radius:999px;
        overflow:hidden; margin:0.4rem 0 0.2rem 0;
    }
    .meter-fill{
        height:100%; border-radius:999px;
        animation:grow 1s cubic-bezier(.22,.9,.36,1);
        background:linear-gradient(90deg, var(--cyan), var(--blue));
    }
    .meter-fill.fake{ background:linear-gradient(90deg, #FCA5A5, var(--red)); }
    .meter-fill.true{ background:linear-gradient(90deg, #6EE7B7, var(--green)); }
    @keyframes grow{ from{ width:0%; } }

    .explain-box{
        margin-top:1.3rem; background:rgba(255,255,255,0.7); border:1px solid var(--border);
        border-radius:14px; padding:0.9rem 1.1rem; font-size:0.88rem; color:#334155;
        text-align:left;
    }

    /* ---------------- MODEL INFO ---------------- */
    .info-grid{ display:grid; grid-template-columns:1fr 1fr; gap:0.9rem; }
    .info-item{
        display:flex; align-items:center; gap:0.7rem;
        background:#F8FAFC; border:1px solid var(--border); border-radius:14px;
        padding:0.85rem 1rem;
    }
    .info-item .ic{ font-size:1.3rem; }
    .info-item .k{ font-size:0.74rem; color:var(--muted); font-weight:700; text-transform:uppercase; letter-spacing:0.04em; }
    .info-item .v{ font-size:0.95rem; color:var(--navy); font-weight:700; }
    .pipeline-strip{
        margin-top:0.9rem; display:flex; align-items:center; justify-content:center;
        gap:0.4rem; flex-wrap:wrap; background:#F1F5F9; border-radius:12px; padding:0.8rem;
        font-family:'JetBrains Mono', monospace; font-size:0.8rem; color:var(--navy); font-weight:600;
    }
    .pipeline-strip .arrow{ color:var(--cyan); font-weight:900; }

    /* ---------------- FEATURES ---------------- */
    .feat-grid{ display:grid; grid-template-columns:1fr 1fr; gap:0.9rem; margin-bottom:1.6rem; }
    .feat-card{
        background:var(--card); border:1px solid var(--border); border-radius:18px;
        padding:1.3rem 1.1rem; text-align:left; transition:all 0.2s ease;
    }
    .feat-card:hover{
        transform:translateY(-4px); box-shadow:0 14px 28px -10px rgba(15,23,42,0.15);
        border-color:#BFDBFE;
    }
    .feat-icon{ font-size:1.6rem; margin-bottom:0.5rem; }
    .feat-title{ font-size:0.96rem; font-weight:800; color:var(--navy); margin-bottom:0.25rem; }
    .feat-desc{ font-size:0.8rem; color:var(--muted); line-height:1.4; }

    /* ---------------- FOOTER ---------------- */
    .footer{
        text-align:center; padding:1.8rem 0 0.6rem 0; border-top:1px solid var(--border); margin-top:1rem;
    }
    .footer .name{ font-weight:800; color:var(--navy); font-size:1.02rem; }
    .footer .tags{ color:var(--muted); font-size:0.82rem; margin:0.25rem 0 0.9rem 0; font-weight:500; }
    .footer .links{ display:flex; justify-content:center; gap:0.9rem; }
    .footer .links a{
        text-decoration:none; font-size:0.85rem; font-weight:700; color:var(--blue);
        background:#EFF6FF; padding:0.4rem 0.85rem; border-radius:999px; border:1px solid #BFDBFE;
        transition:all 0.15s ease;
    }
    .footer .links a:hover{ background:var(--blue); color:white; transform:translateY(-2px); }

    @media (max-width: 640px){
        .info-grid, .feat-grid{ grid-template-columns:1fr; }
        .hero-title{ font-size:1.75rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# HERO SECTION
# ============================================================
st.markdown(
    """
    <div class="hero-wrap">
        <div class="hero-icon">🛰️</div>
        <div class="hero-title">Fake News Detection <span class="grad">using NLP</span></div>
        <div class="hero-sub">AI-Powered News Verification using Natural Language Processing</div>
        <div class="badge-row">
            <div class="badge">✅ 98.7% Accuracy</div>
            <div class="badge blue">⚡ Real-time Inference</div>
        </div>
    </div>
    <div class="divider"></div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# MAIN PREDICTION CARD
# ============================================================
st.markdown(
    """
    <div class="app-card">
        <div class="card-header">
            <div class="icon">📝</div>
            <div>
                <div class="title">Analyze an Article</div>
                <div class="desc">Paste any headline or article body to verify its authenticity</div>
            </div>
        </div>
    """,
    unsafe_allow_html=True,
)

news = st.text_area(
    "Article text",
    value=st.session_state.news_text,
    height=200,
    placeholder="Paste the news title and/or article text here...",
    label_visibility="collapsed",
    key="news_input",
)
st.session_state.news_text = news

char_count = len(news)
word_count = len(news.split()) if news.strip() else 0

st.markdown(
    f"""
    <div class="meta-row">
        <span><span class="tag">{char_count} characters</span> &nbsp; <span class="tag">{word_count} words</span></span>
        <span>TF-IDF + Logistic Regression</span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")
col1, col2, col3 = st.columns([1, 1, 1.6])
with col1:
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state.news_text = ""
        st.session_state.result = None
        st.session_state.has_run = False
        st.rerun()
with col2:
    paste_disabled = False
    st.button("📋 Paste", use_container_width=True, help="Use Ctrl+V inside the text box, then click Analyze")
with col3:
    analyze_clicked = st.button("🚀 Analyze News", type="primary", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)  # close app-card

# ============================================================
# PREDICTION FLOW
# ============================================================
if analyze_clicked:
    if news.strip() == "":
        st.warning("⚠️ Please enter some text first.")
        st.session_state.result = None
    else:
        steps = [
            "🧹 Cleaning text...",
            "🔢 Extracting TF-IDF features...",
            "🧠 Running Logistic Regression...",
            "✨ Finalizing prediction...",
        ]
        progress_area = st.empty()
        for i, step_msg in enumerate(steps):
            progress_area.markdown(
                f"""
                <div class="app-card" style="padding:1.3rem 1.6rem;">
                    <div class="loading-step"><span class="dot"></span> {step_msg}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            time.sleep(0.35)
        progress_area.empty()

        # ---- unchanged backend call ----
        label, prob = predict_news(news)
        st.session_state.result = (label, prob)
        st.session_state.has_run = True

# ============================================================
# RESULT CARD
# ============================================================
if st.session_state.result is not None:
    label, prob = st.session_state.result
    confidence = prob if label == "True" else 1 - prob
    conf_pct = confidence * 100

    if label == "Fake":
        st.markdown(
            f"""
            <div class="result-shell fake">
                <div class="result-icon-wrap fake">🛡️</div>
                <div class="result-label fake">Fake News Detected</div>
                <div class="result-caption">This article shows strong linguistic patterns typical of misinformation.</div>
                <div class="conf-label">Model Confidence</div>
                <div class="conf-num">{conf_pct:.1f}%</div>
                <div class="meter-track"><div class="meter-fill fake" style="width:{conf_pct:.1f}%;"></div></div>
                <div class="explain-box">
                    ⚠️ The classifier's TF-IDF term patterns align more closely with the <b>fake</b> class
                    in the training distribution. Always cross-check with trusted, verified sources.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="result-shell true">
                <div class="result-icon-wrap true">✅</div>
                <div class="result-label true">Verified as True News</div>
                <div class="result-caption">This article's language patterns are consistent with credible reporting.</div>
                <div class="conf-label">Model Confidence</div>
                <div class="conf-num">{conf_pct:.1f}%</div>
                <div class="meter-track"><div class="meter-fill true" style="width:{conf_pct:.1f}%;"></div></div>
                <div class="explain-box">
                    ✅ The classifier's TF-IDF term patterns align more closely with the <b>true</b> class
                    in the training distribution. Confidence reflects model certainty, not absolute truth.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with st.expander("🔬 View raw model output"):
        st.code(f"P(true) = {prob:.4f}\nP(fake) = {1 - prob:.4f}", language="text")

# ============================================================
# MODEL INFORMATION CARD
# ============================================================
st.markdown(
    """
    <div class="app-card">
        <div class="card-header">
            <div class="icon">📊</div>
            <div>
                <div class="title">Model Information</div>
                <div class="desc">Under the hood of this prediction pipeline</div>
            </div>
        </div>
        <div class="info-grid">
            <div class="info-item"><div class="ic">🧠</div><div><div class="k">Model</div><div class="v">Logistic Regression</div></div></div>
            <div class="info-item"><div class="ic">🔢</div><div><div class="k">Vectorizer</div><div class="v">TF-IDF</div></div></div>
            <div class="info-item"><div class="ic">🎯</div><div><div class="k">Accuracy</div><div class="v">98.7%</div></div></div>
            <div class="info-item"><div class="ic">⚡</div><div><div class="k">Inference</div><div class="v">Real-time</div></div></div>
        </div>
        <div class="pipeline-strip">
            <span>Clean Text</span><span class="arrow">→</span>
            <span>TF-IDF</span><span class="arrow">→</span>
            <span>Logistic Regression</span><span class="arrow">→</span>
            <span>Prediction</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# FEATURES SECTION
# ============================================================
st.markdown(
    """
    <div class="feat-grid">
        <div class="feat-card">
            <div class="feat-icon">🤖</div>
            <div class="feat-title">AI Powered</div>
            <div class="feat-desc">Machine learning model trained on real-world news datasets for reliable classification.</div>
        </div>
        <div class="feat-card">
            <div class="feat-icon">🔢</div>
            <div class="feat-title">TF-IDF Vectorization</div>
            <div class="feat-desc">Converts raw text into meaningful numerical features that capture word importance.</div>
        </div>
        <div class="feat-card">
            <div class="feat-icon">⚡</div>
            <div class="feat-title">Fast Prediction</div>
            <div class="feat-desc">Lightweight linear model delivers results in milliseconds — no heavy GPU required.</div>
        </div>
        <div class="feat-card">
            <div class="feat-icon">🎯</div>
            <div class="feat-title">High Accuracy</div>
            <div class="feat-desc">Achieves 98.7% test accuracy, outperforming RNN, LSTM, and GRU baselines.</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# FOOTER
# ============================================================
st.markdown(
    """
    <div class="footer">
        <div class="name">Made by Tarun Chouhan</div>
        <div class="tags">NLP • Machine Learning • Streamlit</div>
        <div class="links">
            <a href="https://github.com/Tarunchouhan926" target="_blank">🐙 GitHub</a>
            <a href="https://linkedin.com/in/tarun-chouhan" target="_blank">💼 LinkedIn</a>
            <a href="mailto:tarunchouhan926@gmail.com">✉️ Email</a>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
