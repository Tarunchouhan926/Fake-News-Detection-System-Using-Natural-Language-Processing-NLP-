import re
import pickle
import streamlit as st

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Fake News Detector | NLP",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- Custom CSS ----------------
st.markdown("""
    <style>
        /* Overall page */
        .main {
            background-color: #f7f9fc;
        }
        /* Hide default Streamlit chrome for a cleaner look */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        /* Header */
        .app-title {
            font-size: 2.3rem;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 0.2rem;
        }
        .app-subtitle {
            font-size: 1.05rem;
            color: #475569;
            margin-bottom: 1.5rem;
        }

        /* Card container */
        .card {
            background-color: #ffffff;
            padding: 1.6rem 1.8rem;
            border-radius: 14px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
            border: 1px solid #e5e9f0;
        }

        /* Result banners */
        .result-real {
            background-color: #ecfdf3;
            border: 1px solid #86efac;
            color: #15803d;
            padding: 1.2rem 1.5rem;
            border-radius: 12px;
            font-size: 1.25rem;
            font-weight: 700;
            text-align: center;
        }
        .result-fake {
            background-color: #fef2f2;
            border: 1px solid #fca5a5;
            color: #b91c1c;
            padding: 1.2rem 1.5rem;
            border-radius: 12px;
            font-size: 1.25rem;
            font-weight: 700;
            text-align: center;
        }

        /* Metric labels */
        .metric-label {
            font-size: 0.85rem;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #0f172a;
        }
        section[data-testid="stSidebar"] * {
            color: #e2e8f0 !important;
        }

        .stButton>button {
            background-color: #1d4ed8;
            color: white;
            font-weight: 600;
            border-radius: 10px;
            padding: 0.6rem 1.5rem;
            border: none;
        }
        .stButton>button:hover {
            background-color: #1e40af;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------- Load Artifacts ----------------
@st.cache_resource
def load_artifacts():
    with open("logistic_regression_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)
    return model, tfidf

model, tfidf = load_artifacts()

# ---------------- Text Cleaning (same as training) ----------------
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
    label = "Real" if prob > 0.5 else "Fake"
    return label, float(prob)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("### 📰 About This Project")
    st.write(
        "An NLP-based system that classifies news articles as **Real** or **Fake** "
        "using TF-IDF feature extraction and a Logistic Regression classifier."
    )
    st.markdown("---")
    st.markdown("**Model:** TF-IDF + Logistic Regression")
    st.markdown("**Test Accuracy:** 98.7%")
    st.markdown("**Compared Against:** SimpleRNN, LSTM, GRU")
    st.markdown("---")
    st.markdown("**Built by:** Tarun Chouhan")
    st.markdown(
        """
        <div style="display: flex; gap: 10px; margin-top: 8px;">
            <a href="https://github.com/Tarunchouhan926/Fake-News-Detection-System-Using-Natural-Language-Processing-NLP-" target="_blank" style="text-decoration:none;">
                <img src="https://img.shields.io/badge/GitHub-Repo-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub">
            </a>
            <a href="https://linkedin.com/in/tarun-chouhan" target="_blank" style="text-decoration:none;">
                <img src="https://img.shields.io/badge/LinkedIn-Profile-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn">
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------- Header ----------------
st.markdown('<div class="app-title">📰 Fake News Detection System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Paste any news article below to check whether it\'s likely Real or Fake, '
    'powered by NLP and Machine Learning.</div>',
    unsafe_allow_html=True,
)

# ---------------- Main Input Card ----------------
col1, col2 = st.columns([2, 1], gap="large")

with col1:
    card1 = st.container(border=True)
    with card1:
        news = st.text_area(
            "News Article Text",
            height=220,
            placeholder="Paste the news title and/or article text here...",
            label_visibility="collapsed",
        )
        check = st.button("🔍 Analyze Article", use_container_width=False)

with col2:
    card2 = st.container(border=True)
    with card2:
        st.markdown('<div class="metric-label">How it works</div>', unsafe_allow_html=True)
        st.markdown(
            """
            1. Text is cleaned and normalized  
            2. Converted into TF-IDF features  
            3. Classified by a trained Logistic Regression model  
            4. Confidence score is calculated
            """
        )

# ---------------- Result Section ----------------
if check:
    if news.strip() == "":
        st.warning("Please enter some article text first.")
    else:
        label, prob = predict_news(news)
        confidence = prob if label == "Real" else 1 - prob

        st.markdown("### Result")

        result_col, gauge_col = st.columns([1, 2], gap="large")

        with result_col:
            css_class = "result-real" if label == "Real" else "result-fake"
            icon = "✅" if label == "Real" else "🚫"
            st.markdown(
                f'<div class="{css_class}">{icon} {label} News<br>'
                f'<span style="font-size:0.95rem; font-weight:500;">Confidence: {confidence:.1%}</span></div>',
                unsafe_allow_html=True,
            )

        with gauge_col:
            st.markdown('<div class="metric-label">Confidence Level</div>', unsafe_allow_html=True)
            st.progress(confidence)
            m1, m2 = st.columns(2)
            m1.metric("Predicted Class", label)
            m2.metric("Confidence Score", f"{confidence:.2%}")
