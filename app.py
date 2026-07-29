import re
import pickle
import streamlit as st

st.set_page_config(page_title="Fake News Classifier", page_icon="📰", layout="centered")

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

# ---------------- Streamlit UI ----------------
st.title("📰 Fake News Classifier")
st.write("Powered by TF-IDF + Logistic Regression (98.7% test accuracy)")

news = st.text_area("Enter a news article:", height=200, placeholder="Paste the news title and/or article text here...")

if st.button("Check News"):
    if news.strip() == "":
        st.warning("Please enter some text first.")
    else:
        label, prob = predict_news(news)
        confidence = prob if label == "True" else 1 - prob

        if label == "True":
            st.success(f"**{label}** (confidence: {confidence:.2%})")
        else:
            st.error(f"**{label}** (confidence: {confidence:.2%})")

        st.progress(float(prob))
        st.caption(f"Raw model output (probability of true): {prob:.4f}")