"""
Fake News Detection using NLP
Project by: Tarun Chouhan
Model: TF-IDF + Logistic Regression (Test Accuracy: 98.7%)
--------------------------------------------------------------
Note: This is the same backend logic as the original app.
Only the layout/design is simplified for a clean project-style UI.
"""

import re
import pickle
import streamlit as st

# ----------------------------------------------------
# Page config
# ----------------------------------------------------
st.set_page_config(page_title="Fake News Detection - NLP Project", page_icon="📰", layout="centered")

# ----------------------------------------------------
# Load model + vectorizer (unchanged)
# ----------------------------------------------------
@st.cache_resource
def load_artifacts():
    with open("logistic_regression_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)
    return model, tfidf


model, tfidf = load_artifacts()


# ----------------------------------------------------
# Same cleaning function used during training (unchanged)
# ----------------------------------------------------
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


# ----------------------------------------------------
# Minimal styling (just to make it look neat, nothing flashy)
# ----------------------------------------------------
st.markdown(
    """
    <style>
    .block-container{ max-width:750px; padding-top:2rem; }
    h1{ font-size:1.9rem !important; }
    .project-info{
        background:#f0f2f6; padding:0.8rem 1rem; border-radius:8px;
        font-size:0.9rem; margin-bottom:1rem; border-left:4px solid #4C6EF5;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------
# Title + basic project info
# ----------------------------------------------------
st.title("📰 Fake News Detection System Using NLP")
st.caption("A Machine Learning mini-project using TF-IDF and Logistic Regression")

st.markdown(
    """
    <div class="project-info">
    <b>Project by:</b> Tarun Chouhan<br>
    <b>Model Used:</b> Logistic Regression &nbsp; | &nbsp; <b>Vectorizer:</b> TF-IDF<br>
    <b>Test Accuracy:</b> 98.7%
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("---")

# ----------------------------------------------------
# Input section
# ----------------------------------------------------
st.subheader("Enter a news article")
news = st.text_area(
    "Paste the news title and/or article text below:",
    height=200,
    placeholder="Paste the news title and/or article text here...",
)

col1, col2 = st.columns(2)
with col1:
    st.caption(f"Character count: {len(news)}")
with col2:
    st.caption(f"Word count: {len(news.split()) if news.strip() else 0}")

check = st.button("Check News", use_container_width=True)

# ----------------------------------------------------
# Prediction + result
# ----------------------------------------------------
if check:
    if news.strip() == "":
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Analyzing the article..."):
            label, prob = predict_news(news)
        confidence = prob if label == "True" else 1 - prob

        st.write("### Result")
        if label == "True":
            st.success(f"**Prediction: {label} News** (confidence: {confidence:.2%})")
        else:
            st.error(f"**Prediction: {label} News** (confidence: {confidence:.2%})")

        st.progress(float(confidence))
        st.caption(f"Raw model output (probability of true): {prob:.4f}")

        with st.expander("How this works"):
            st.write(
                """
                1. The input text is cleaned (lowercased, HTML tags and special characters removed).
                2. The cleaned text is converted into numerical features using **TF-IDF**.
                3. A trained **Logistic Regression** model predicts the probability that the
                   article is *True*.
                4. If the probability is greater than 0.5, the article is classified as **True**,
                   otherwise it is classified as **Fake**.
                """
            )

st.write("---")

# ----------------------------------------------------
# Footer
# ----------------------------------------------------
st.caption("Made by Tarun Chouhan | NLP • Machine Learning • Streamlit")
st.caption("GitHub: github.com/Tarunchouhan926  |  LinkedIn: linkedin.com/in/tarun-chouhan")
