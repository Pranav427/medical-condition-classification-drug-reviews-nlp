import html
import pickle
import re
import time
from pathlib import Path

import nltk
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "model.pkl"
VECTORIZER_PATH = APP_DIR / "vectorizer.pkl"
LABEL_ENCODER_PATH = APP_DIR / "label_encoder.pkl"

GITHUB_URL = "https://github.com/Pranav427/medical-condition-classification-drug-reviews-nlp"
PORTFOLIO_URL = "https://portfolio-self-one-10.vercel.app"

SUPPORTED_CONDITIONS = ("Depression", "Diabetes, Type 2", "High Blood Pressure")

SAMPLE_REVIEWS = {
    "Depression": (
        "I started this medication after months of low mood, anxiety, and poor sleep. "
        "After a few weeks I felt more stable, had better energy, and could manage "
        "daily activities again, although the first few days were difficult."
    ),
    "Diabetes, Type 2": (
        "My blood sugar levels improved after taking this medicine for several weeks. "
        "I noticed reduced appetite and better glucose readings, but nausea and "
        "stomach discomfort made the treatment harder to continue."
    ),
    "High Blood Pressure": (
        "This blood pressure medicine helped bring my readings closer to normal. "
        "I felt less pressure and fewer headaches, but sometimes experienced dizziness "
        "and tiredness after taking the dose."
    ),
}

MODEL_COMPARISON = pd.DataFrame(
    {
        "Model": [
            "Linear SVM",
            "SGD Classifier",
            "Logistic Regression",
            "Naive Bayes",
        ],
        "Macro F1": [0.945971, 0.941714, 0.936224, 0.919484],
        "Accuracy": [0.961635, 0.958049, 0.953030, 0.941198],
    }
)

CLASSIFICATION_REPORT = pd.DataFrame(
    {
        "Class": ["Depression", "Diabetes, Type 2", "High Blood Pressure"],
        "Precision": [0.97, 0.94, 0.94],
        "Recall": [0.98, 0.95, 0.90],
        "F1-score": [0.98, 0.94, 0.92],
        "Support": [1814, 511, 464],
    }
)

CONFUSION_MATRIX = pd.DataFrame(
    [[1780, 14, 20], [14, 486, 11], [34, 13, 417]],
    index=SUPPORTED_CONDITIONS,
    columns=SUPPORTED_CONDITIONS,
)

DATASET_COUNTS = pd.DataFrame(
    {
        "Condition": ["Depression", "Diabetes, Type 2", "High Blood Pressure"],
        "Reviews": [9069, 2554, 2321],
    }
)

RATING_DISTRIBUTION = pd.DataFrame(
    {
        "Rating": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "Reviews": [1802, 645, 582, 458, 756, 680, 995, 1795, 2468, 3763],
    }
)

REVIEW_LENGTH_DISTRIBUTION = pd.DataFrame(
    {
        "Review length": [
            "0-25",
            "26-50",
            "51-75",
            "76-100",
            "101-150",
            "151-250",
            "251-500",
            "500+",
        ],
        "Reviews": [1566, 2333, 2410, 2182, 5229, 202, 18, 4],
    }
)

SENTIMENT_DISTRIBUTION = pd.DataFrame(
    {
        "Sentiment": ["Positive", "Negative", "Neutral"],
        "Reviews": [9021, 3029, 1894],
    }
)

TOP_TERMS = pd.DataFrame(
    {
        "Term": ["day", "effect", "side", "week", "year", "taking", "depression", "feel"],
        "Frequency": [7373, 6811, 6227, 6126, 5897, 5738, 5624, 5424],
    }
)


def ensure_nltk_resource(resource_path: str, package_name: str) -> None:
    try:
        nltk.data.find(resource_path)
    except LookupError:
        nltk.download(package_name, quiet=True)


@st.cache_resource(show_spinner=False)
def load_nlp_tools():
    ensure_nltk_resource("corpora/stopwords", "stopwords")
    ensure_nltk_resource("corpora/wordnet", "wordnet")
    ensure_nltk_resource("sentiment/vader_lexicon.zip", "vader_lexicon")

    return {
        "stop_words": set(stopwords.words("english")),
        "lemmatizer": WordNetLemmatizer(),
        "sentiment_analyzer": SentimentIntensityAnalyzer(),
    }


@st.cache_resource(show_spinner="Loading model artifacts...")
def load_artifacts():
    required_files = [MODEL_PATH, VECTORIZER_PATH, LABEL_ENCODER_PATH]
    missing_files = [path.name for path in required_files if not path.exists()]
    if missing_files:
        raise FileNotFoundError(
            "Missing required model artifact(s): " + ", ".join(missing_files)
        )

    with MODEL_PATH.open("rb") as f:
        model = pickle.load(f)
    with VECTORIZER_PATH.open("rb") as f:
        vectorizer = pickle.load(f)
    with LABEL_ENCODER_PATH.open("rb") as f:
        label_encoder = pickle.load(f)

    return model, vectorizer, label_encoder


def clean_review(text: str, stop_words: set[str], lemmatizer: WordNetLemmatizer) -> str:
    text = html.unescape(html.unescape(text)).lower()
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"https\S+|www\.\S+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    words = [lemmatizer.lemmatize(word) for word in words]
    return " ".join(words)


def sentiment_label(score: float) -> str:
    if score >= 0.05:
        return "Positive"
    if score <= -0.05:
        return "Negative"
    return "Neutral"


def prediction_strength(model, review_vector) -> tuple[str | None, float | None]:
    if not hasattr(model, "decision_function"):
        return None, None

    scores = model.decision_function(review_vector)
    if scores.ndim == 1:
        margin = abs(float(scores[0]))
    else:
        sorted_scores = sorted(scores[0], reverse=True)
        margin = float(sorted_scores[0] - sorted_scores[1])

    if margin >= 2:
        return "High", 0.9
    if margin >= 0.75:
        return "Moderate", 0.62
    return "Low", 0.34


def set_sample_review(condition: str) -> None:
    st.session_state.review_text = SAMPLE_REVIEWS[condition]


def clear_review() -> None:
    st.session_state.review_text = ""


def open_case_study() -> None:
    st.session_state.page = "Technical Case Study"


def apply_page_styles() -> None:
    st.markdown(
        """
        <style>
        html {
            scroll-behavior: smooth;
        }
        .block-container {
            max-width: 1120px;
            padding-top: 3.4rem;
            padding-bottom: 3rem;
        }
        section[data-testid="stSidebar"] {
            width: 16.5rem !important;
        }
        div[data-testid="stMetric"] {
            border: 1px solid rgba(128, 128, 128, 0.20);
            border-radius: 8px;
            padding: 0.75rem 0.9rem;
            background: rgba(128, 128, 128, 0.055);
            transition: transform 160ms ease, border-color 160ms ease, box-shadow 160ms ease;
        }
        div[data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            border-color: rgba(122, 162, 255, 0.42);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.18);
        }
        div[data-testid="stMetricValue"] {
            font-size: 1.55rem;
        }
        div[data-testid="stMetricLabel"] {
            font-size: 0.82rem;
        }
        div[data-testid="stTextArea"] textarea {
            line-height: 1.55;
            min-height: 170px;
        }
        .app-hero {
            padding: 0.25rem 0 1rem 0;
            margin-bottom: 0.7rem;
        }
        .app-hero h1 { font-size: 2.25rem; line-height: 1.12; margin-bottom: 0.55rem; }
        .app-hero p { font-size: 1.02rem; max-width: 760px; }
        h1 a, h2 a, h3 a {
            display: none !important;
        }
        .trust-chip {
            display: inline-block;
            border: 1px solid rgba(128, 128, 128, 0.22);
            border-radius: 999px;
            padding: 0.28rem 0.68rem;
            margin: 0.15rem 0.2rem 0.15rem 0;
            background: rgba(128, 128, 128, 0.045);
            font-size: 0.86rem;
            color: rgba(245, 245, 245, 0.86);
        }
        .hero-metrics {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.65rem;
            margin-top: 0.9rem;
            max-width: 850px;
            animation: softFadeIn 420ms ease-out;
        }
        .hero-metric {
            border: 1px solid rgba(128, 128, 128, 0.18);
            border-radius: 8px;
            padding: 0.62rem 0.72rem;
            background: rgba(128, 128, 128, 0.04);
            transition: transform 160ms ease, border-color 160ms ease, box-shadow 160ms ease;
        }
        .hero-metric:hover {
            transform: translateY(-2px);
            border-color: rgba(122, 162, 255, 0.42);
            box-shadow: 0 8px 22px rgba(0, 0, 0, 0.16);
        }
        .hero-metric-value {
            font-size: 1rem;
            font-weight: 750;
            line-height: 1.2;
        }
        .hero-metric-label {
            color: rgba(160, 160, 160, 0.95);
            font-size: 0.76rem;
            margin-top: 0.12rem;
        }
        .eyebrow {
            color: #7aa2ff;
            font-size: 0.85rem;
            font-weight: 700;
            letter-spacing: 0.02rem;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }
        .muted {
            color: rgba(128, 128, 128, 0.95);
        }
        .result-card {
            border: 1px solid rgba(128, 128, 128, 0.20);
            border-radius: 10px;
            padding: 1.15rem 1.25rem;
            background: rgba(128, 128, 128, 0.045);
            animation: softFadeIn 360ms ease-out;
            transition: transform 160ms ease, border-color 160ms ease, box-shadow 160ms ease;
        }
        .result-card:hover {
            transform: translateY(-2px);
            border-color: rgba(122, 162, 255, 0.42);
            box-shadow: 0 10px 28px rgba(0, 0, 0, 0.20);
        }
        .result-label {
            color: rgba(128, 128, 128, 0.95);
            font-size: 0.9rem;
            margin-bottom: 0.1rem;
        }
        .result-value {
            font-size: 1.9rem;
            font-weight: 750;
            line-height: 1.15;
            margin-bottom: 0.8rem;
        }
        .section-card {
            border: 1px solid rgba(128, 128, 128, 0.18);
            border-radius: 10px;
            padding: 1rem 1.1rem;
            background: rgba(128, 128, 128, 0.035);
            height: 100%;
        }
        div[data-testid="stVerticalBlockBorderWrapper"] {
            transition: transform 160ms ease, border-color 160ms ease, box-shadow 160ms ease;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:hover {
            transform: translateY(-1px);
            border-color: rgba(122, 162, 255, 0.30);
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.14);
        }
        .stButton button,
        .stLinkButton a {
            transition: transform 150ms ease, border-color 150ms ease, box-shadow 150ms ease;
            border-radius: 8px;
        }
        .stButton button:hover,
        .stLinkButton a:hover {
            transform: translateY(-1px);
            box-shadow: 0 8px 22px rgba(0, 0, 0, 0.16);
        }
        .roadmap {
            border: 1px solid rgba(128, 128, 128, 0.18);
            border-radius: 10px;
            padding: 1rem 1.1rem;
            background: rgba(128, 128, 128, 0.035);
            line-height: 1.9;
        }
        .roadmap-step {
            display: inline-block;
            border: 1px solid rgba(122, 162, 255, 0.34);
            border-radius: 999px;
            padding: 0.24rem 0.62rem;
            margin: 0.18rem;
            background: rgba(122, 162, 255, 0.08);
            font-size: 0.92rem;
        }
        .workflow {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            gap: 0.55rem;
            border: 1px solid rgba(128, 128, 128, 0.18);
            border-radius: 10px;
            padding: 1rem 1.1rem;
            background: rgba(128, 128, 128, 0.035);
            margin: 0.8rem 0 1.2rem 0;
        }
        .workflow-node {
            border: 1px solid rgba(122, 162, 255, 0.34);
            border-radius: 8px;
            padding: 0.56rem 0.78rem;
            background: rgba(122, 162, 255, 0.08);
            font-size: 0.88rem;
            font-weight: 650;
            white-space: nowrap;
        }
        .workflow-arrow {
            color: rgba(245, 245, 245, 0.62);
            font-size: 1rem;
        }
        .takeaway-card {
            border: 1px solid rgba(122, 162, 255, 0.28);
            border-radius: 10px;
            padding: 1rem 1.1rem;
            background: rgba(122, 162, 255, 0.055);
        }
        @keyframes softFadeIn {
            from { opacity: 0; transform: translateY(8px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @media (max-width: 760px) {
            .hero-metrics {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .app-hero h1 {
                font-size: 1.85rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_bar_chart(
    data: pd.DataFrame,
    x: str,
    y: str,
    *,
    horizontal: bool = False,
    height: float = 3.1,
    rotation: int = 0,
    grouped_columns: list[str] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, height), facecolor="#0e1117")
    ax.set_facecolor("#0e1117")

    if grouped_columns:
        positions = np.arange(len(data[x]))
        width = 0.34
        colors = ["#0f73c9", "#78bdf2"]
        for index, column in enumerate(grouped_columns):
            offset = (index - (len(grouped_columns) - 1) / 2) * width
            ax.bar(
                positions + offset,
                data[column],
                width=width,
                color=colors[index],
                label=column,
            )
        ax.set_xticks(positions)
        ax.set_xticklabels(data[x])
        ax.set_ylim(0.88, 0.98)
        ax.legend(frameon=False, labelcolor="#f5f5f5", loc="lower left")
    elif horizontal:
        ax.barh(data[x], data[y], color="#78bdf2")
        ax.invert_yaxis()
    else:
        ax.bar(data[x], data[y], color="#78bdf2")

    ax.grid(axis="y", color="#30343d", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#d7d9df", labelsize=9)
    ax.tick_params(axis="x", rotation=rotation)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def render_workflow(steps: list[str]) -> None:
    nodes = []
    for index, step in enumerate(steps):
        nodes.append(f'<span class="workflow-node">{step}</span>')
        if index < len(steps) - 1:
            nodes.append('<span class="workflow-arrow">→</span>')
    st.markdown(f'<div class="workflow">{"".join(nodes)}</div>', unsafe_allow_html=True)


def render_sidebar() -> str:
    with st.sidebar:
        st.title("Patient NLP")
        page = st.radio(
            "Choose page",
            ["AI Prediction System", "Technical Case Study"],
            label_visibility="collapsed",
            key="page",
        )
        st.divider()
        if GITHUB_URL:
            st.link_button("GitHub", GITHUB_URL, use_container_width=True)
        if PORTFOLIO_URL:
            st.link_button("Portfolio", PORTFOLIO_URL, use_container_width=True)
        st.divider()
        st.caption("Applied NLP Project")
        st.write("✓ 161K Drug Reviews")
        st.write("✓ TF-IDF Feature Engineering")
        st.write("✓ Linear SVM Classifier")
        st.write("✓ 96.16% Test Accuracy")
        st.write("✓ Streamlit Deployment")
    return page


def render_metric_row() -> None:
    metric_cols = st.columns(4)
    metric_cols[0].metric("Model", "Linear SVM")
    metric_cols[1].metric("Accuracy", "96.16%")
    metric_cols[2].metric("Macro F1", "94.60%")
    metric_cols[3].metric("Classes", "3")


def render_prediction_result(
    condition: str,
    sentiment: str,
    sentiment_score: float,
    strength: str | None,
    strength_score: float | None,
    cleaned_review: str,
) -> None:
    st.subheader("Prediction Result")

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">Predicted condition</div>
            <div class="result-value">{html.escape(condition)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    result_cols = st.columns(3)
    result_cols[0].metric("Sentiment", sentiment)
    result_cols[1].metric("Sentiment score", f"{sentiment_score:.2f}")
    result_cols[2].metric("Match strength", strength or "Unavailable")

    if strength_score is not None:
        st.progress(strength_score, text="Model match strength")

    if strength == "Low":
        st.warning(
            "Low match strength means the review may not closely fit the three "
            "supported categories. Treat the output as a directional signal."
        )
    else:
        st.success("The review has a clear match among the supported categories.")

    with st.expander("Technical details"):
        st.write("Preprocessed model input")
        st.code(cleaned_review or "No cleaned text available.", language="text")
        st.caption(
            "Match strength is derived from the Linear SVM decision margin. "
            "It is not a calibrated probability."
        )


def render_processing_sequence() -> None:
    steps = [
        "Validating input...",
        "Cleaning review...",
        "Removing stopwords...",
        "Lemmatizing text...",
        "TF-IDF vectorization...",
        "Running Linear SVM...",
        "Calculating sentiment...",
        "Prediction ready",
    ]
    progress_placeholder = st.empty()
    status_placeholder = st.empty()

    for index, step in enumerate(steps, start=1):
        progress_placeholder.progress(index / len(steps), text=step)
        status_placeholder.caption(f"✓ {step}")
        time.sleep(0.18)

    progress_placeholder.empty()
    status_placeholder.empty()


def render_prediction_page(model, tfidf, label_encoder, nlp_tools) -> None:
    st.markdown(
        """
        <div class="app-hero">
            <div class="eyebrow">AI Prediction System</div>
            <h1>Patient Condition Classification using NLP</h1>
            <p class="muted">
            An end-to-end Natural Language Processing application that classifies patient
            drug reviews into medical condition categories using TF-IDF feature engineering
            and a Linear SVM classifier.
            </p>
            <span class="trust-chip">TF-IDF NLP</span>
            <span class="trust-chip">Linear SVM</span>
            <span class="trust-chip">96.16% accuracy</span>
            <span class="trust-chip">3 supported classes</span>
            <div class="hero-metrics">
                <div class="hero-metric">
                    <div class="hero-metric-value">161,297</div>
                    <div class="hero-metric-label">Drug Reviews</div>
                </div>
                <div class="hero-metric">
                    <div class="hero-metric-value">13,944</div>
                    <div class="hero-metric-label">Training Samples</div>
                </div>
                <div class="hero-metric">
                    <div class="hero-metric-value">96.16%</div>
                    <div class="hero-metric-label">Accuracy</div>
                </div>
                <div class="hero-metric">
                    <div class="hero-metric-value">94.60%</div>
                    <div class="hero-metric-label">Macro F1</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if "review_text" not in st.session_state:
        st.session_state.review_text = ""

    with st.container(border=True):
        st.subheader("Review Analyzer")
        st.caption("Choose a clean sample review or paste a patient drug review.")

        sample_cols = st.columns(3)
        for index, condition in enumerate(SUPPORTED_CONDITIONS):
            sample_cols[index].button(
                condition,
                use_container_width=True,
                on_click=set_sample_review,
                args=(condition,),
            )

        user_review = st.text_area(
            "Patient review",
            key="review_text",
            placeholder="Example: This medicine helped control my blood pressure and I feel stable now.",
        )

        word_count = len(user_review.split())
        detail_cols = st.columns([1, 1, 2])
        detail_cols[0].caption(f"Words: {word_count}")
        detail_cols[1].caption(f"Characters: {len(user_review)}")
        detail_cols[2].caption(
            "Supported outputs: Depression, Diabetes Type 2, High Blood Pressure"
        )

        action_cols = st.columns([3, 1])
        predict_btn = action_cols[0].button(
            "Analyze Review",
            type="primary",
            use_container_width=True,
        )
        action_cols[1].button("Clear", use_container_width=True, on_click=clear_review)

        if not user_review.strip():
            st.info("Enter a review or select a sample to generate a prediction.")

        if predict_btn:
            if not user_review.strip():
                st.warning("Please enter a patient review before predicting.")
            elif word_count < 5:
                st.warning(
                    "Please enter a more detailed review so the model has enough context."
                )
            else:
                render_processing_sequence()
                with st.spinner("Analyzing review text..."):
                    cleaned_review = clean_review(
                        user_review,
                        nlp_tools["stop_words"],
                        nlp_tools["lemmatizer"],
                    )

                    if not cleaned_review:
                        st.warning(
                            "The review does not contain enough meaningful words after cleaning."
                        )
                        st.stop()

                    review_vec = tfidf.transform([cleaned_review])
                    pred = model.predict(review_vec)
                    condition = label_encoder.inverse_transform(pred)[0]
                    sentiment_score = nlp_tools["sentiment_analyzer"].polarity_scores(
                        user_review
                    )["compound"]
                    sentiment = sentiment_label(sentiment_score)
                    strength, strength_score = prediction_strength(model, review_vec)

                st.divider()
                render_prediction_result(
                    condition,
                    sentiment,
                    sentiment_score,
                    strength,
                    strength_score,
                    cleaned_review,
                )

    st.divider()
    info_cols = st.columns([1, 1, 1])
    with info_cols[0]:
        with st.container(border=True):
            st.subheader("Model Summary")
            st.write("Interpretable text classifier using TF-IDF unigram/bigram features with Linear SVM.")
            st.caption("Accuracy: 96.16% | Macro F1: 94.60%")
            st.link_button("View GitHub Repository", GITHUB_URL, use_container_width=True)
    with info_cols[1]:
        with st.container(border=True):
            st.subheader("Technical Case Study")
            st.write("Explore the dataset, NLP pipeline, model comparison, evaluation, and deployment architecture.")
            st.button(
                "Open Technical Case Study",
                use_container_width=True,
                on_click=open_case_study,
            )
    with info_cols[2]:
        with st.container(border=True):
            st.subheader("Responsible Use")
            st.write(
                "Only predicts among three trained categories. Match strength is not "
                "a calibrated probability."
            )
            st.caption("Educational project only. Not medical advice.")


def render_case_section(title: str, body: str) -> None:
    st.subheader(title)
    st.write(body)


def render_roadmap() -> None:
    steps = [
        "Business Problem",
        "Dataset",
        "EDA",
        "Cleaning",
        "NLP Pipeline",
        "TF-IDF",
        "Model Selection",
        "Tuning",
        "Evaluation",
        "Deployment",
    ]
    html_steps = " ".join(f'<span class="roadmap-step">{step}</span>' for step in steps)
    st.markdown(f'<div class="roadmap">{html_steps}</div>', unsafe_allow_html=True)


def render_case_study_page() -> None:
    st.markdown(
        """
        <div class="app-hero">
            <div class="eyebrow">Technical Case Study</div>
            <h1>Engineering Journey</h1>
            <p class="muted">
            A concise walkthrough of the data, NLP pipeline, model selection,
            evaluation, deployment, and product decisions behind the application.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    overview_cols = st.columns(4)
    overview_cols[0].metric("Raw records", "161,297")
    overview_cols[1].metric("Training subset", "13,944")
    overview_cols[2].metric("Best CV F1", "94.87%")
    overview_cols[3].metric("Test F1", "94.60%")

    st.subheader("Engineering Roadmap")
    render_roadmap()

    tabs = st.tabs(["Problem & Data", "NLP Pipeline", "Modeling", "Evaluation", "Deployment"])

    with tabs[0]:
        left, right = st.columns([1, 1])
        with left:
            render_case_section(
                "Business Problem",
                "Patient drug reviews are unstructured. The product goal is to organize "
                "review text into clinically relevant condition categories for analysis."
            )
            render_bar_chart(DATASET_COUNTS, "Condition", "Reviews", horizontal=True)
        with right:
            render_case_section(
                "EDA Focus",
                "The analysis examined class balance, review length, rating behavior, "
                "frequent terms, and sentiment distribution across the selected classes."
            )
            render_bar_chart(REVIEW_LENGTH_DISTRIBUTION, "Review length", "Reviews")
        chart_cols = st.columns(2)
        with chart_cols[0]:
            render_bar_chart(RATING_DISTRIBUTION, "Rating", "Reviews")
        with chart_cols[1]:
            render_bar_chart(SENTIMENT_DISTRIBUTION, "Sentiment", "Reviews")

    with tabs[1]:
        render_case_section(
            "Cleaning and Feature Engineering",
            "The deployed app mirrors notebook preprocessing: HTML cleanup, URL removal, "
            "punctuation removal, stopword filtering, and lemmatization before TF-IDF."
        )
        render_workflow(["Review", "Clean Text", "Tokens", "Lemmas", "TF-IDF", "Features"])
        st.subheader("Common Terms After Cleaning")
        render_bar_chart(TOP_TERMS, "Term", "Frequency", horizontal=True)

    with tabs[2]:
        render_case_section(
            "Model Selection",
            "Four classical ML models were compared. Linear SVM gave the best balance of "
            "macro F1, accuracy, speed, and deployment simplicity."
        )
        render_bar_chart(
            MODEL_COMPARISON,
            "Model",
            "Accuracy",
            height=3.4,
            rotation=0,
            grouped_columns=["Macro F1", "Accuracy"],
        )
        st.dataframe(
            MODEL_COMPARISON.style.format({"Macro F1": "{:.3f}", "Accuracy": "{:.3f}"}),
            use_container_width=True,
            hide_index=True,
        )
        tune_cols = st.columns(2)
        tune_cols[0].metric("Best C", "1")
        tune_cols[1].metric("Best CV Macro F1", "94.87%")

    with tabs[3]:
        render_case_section(
            "Model Evaluation",
            "The final model achieved 96.16% test accuracy and 94.60% macro F1. "
            "Macro F1 is important because the selected classes are imbalanced."
        )
        st.dataframe(
            CLASSIFICATION_REPORT.style.format(
                {"Precision": "{:.2f}", "Recall": "{:.2f}", "F1-score": "{:.2f}"}
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.subheader("Confusion Matrix")
        st.dataframe(CONFUSION_MATRIX, use_container_width=True)

    with tabs[4]:
        render_case_section(
            "Prediction Workflow",
            "The runtime app loads saved artifacts, validates input, applies preprocessing, "
            "runs TF-IDF vectorization, predicts with Linear SVM, and displays results."
        )
        render_workflow(
            [
                "User",
                "Streamlit UI",
                "Input Validation",
                "Text Cleaning",
                "TF-IDF",
                "Linear SVM",
                "Results Dashboard",
            ]
        )
        st.caption("Sentiment is calculated in parallel as supporting context.")
        deployment_cols = st.columns(2)
        with deployment_cols[0]:
            st.subheader("Application Architecture")
            st.code(
                """streamlit_app.py
├── model.pkl
├── vectorizer.pkl
├── label_encoder.pkl
└── runtime UI + case study""",
                language="text",
            )
        with deployment_cols[1]:
            st.subheader("Engineering Maturity")
            st.write("- Cached artifact loading")
            st.write("- Deployment-safe paths")
            st.write("- Input validation and empty states")
            st.write("- Clear limitations and medical disclaimer")

    st.divider()
    st.markdown(
        """
        <div class="takeaway-card">
            <strong>Project Impact</strong><br>
            Demonstrates end-to-end NLP, practical model evaluation, deployment-ready
            engineering, and product judgment for a healthcare-adjacent AI workflow.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    final_cols = st.columns(3)
    with final_cols[0]:
        with st.container(border=True):
            st.subheader("Challenges")
            st.write("Class imbalance, preprocessing consistency, dependency compatibility, and responsible AI framing.")
    with final_cols[1]:
        with st.container(border=True):
            st.subheader("Lessons Learned")
            st.write("A strong ML project needs reliable inference, clear UX, documentation, and honest limitations.")
    with final_cols[2]:
        with st.container(border=True):
            st.subheader("Future Improvements")
            st.write("Single Pipeline artifact, tests, model card, calibrated probabilities, and error analysis.")


st.set_page_config(
    page_title="Patient Condition Classification",
    page_icon="PC",
    layout="wide",
)
apply_page_styles()

page = render_sidebar()

try:
    with st.spinner("Preparing NLP resources..."):
        nlp_tools = load_nlp_tools()
        model, tfidf, label_encoder = load_artifacts()
except Exception as exc:
    st.error("The application could not start because a required resource is unavailable.")
    st.exception(exc)
    st.stop()

if page == "AI Prediction System":
    render_prediction_page(model, tfidf, label_encoder, nlp_tools)
else:
    render_case_study_page()
