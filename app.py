import streamlit as st
import pickle

# --- Handle NLTK safely ---
import nltk
try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
except:
    nltk.download('vader_lexicon')
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

# --- Page configuration ---
st.set_page_config(
    page_title="Patient Condition Classification",
    page_icon="🩺",
    layout="centered"
)

# --- Load saved artifacts ---
with open("linear_svm_tuned.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# --- Header ---
st.markdown(
    "<h2 style='text-align: center;'>🩺 Patient Condition Classification</h2>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center; color: gray;'>"
    "Predict the medical condition and sentiment from a patient drug review"
    "</p>",
    unsafe_allow_html=True
)

st.markdown("---")

# --- User Input Section ---
st.subheader("📝 Enter Patient Review")

user_review = st.text_area(
    "",
    height=160,
    placeholder="Example: This medicine helped control my blood pressure and I feel stable now."
)

# --- Prediction Button ---
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    predict_btn = st.button("🔍 Predict")

# --- Prediction Logic ---
if predict_btn:
    if user_review.strip() == "":
        st.warning("⚠️ Please enter a patient review before predicting.")
    else:
        # Condition prediction
        review_vec = tfidf.transform([user_review])
        pred = model.predict(review_vec)
        condition = le.inverse_transform(pred)[0]

        # Sentiment prediction
        sentiment_score = sia.polarity_scores(user_review)['compound']
        if sentiment_score >= 0.05:
            sentiment = "Positive 😊"
        elif sentiment_score <= -0.05:
            sentiment = "Negative 😟"
        else:
            sentiment = "Neutral 😐"

        st.markdown("---")

        # --- Results ---
        st.subheader("📊 Prediction Result")

        st.success(f"**Predicted Condition:** {condition}")
        st.info(f"**Review Sentiment:** {sentiment}")

# --- Footer ---
st.markdown("---")
st.caption(
    "ℹ️ Project Note: The predictions are generated using a trained machine learning "
    "model on historical drug review data to demonstrate text classification "
    "and sentiment analysis techniques."
)

