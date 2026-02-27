"""
Streamlit App for Sentiment Analysis
"""

import sys
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

sys.path.append('src')

import streamlit as st
from data_preprocessing import TextPreprocessor
from feature_engineering import prepare_train_test_split
from train_models import SentimentModelTrainer
from data_validator import DataValidator


SENTIMENT_MAP = {0: 'Negative', 1: 'Positive', 2: 'Neutral'}
SENTIMENT_EMOJI = {0: '😞', 1: '😊', 2: '😐'}


def create_demo_dataset(size='medium'):
    sizes = {'small': 20, 'medium': 40, 'large': 80}
    multiplier = sizes.get(size, 40)

    positive_texts = [
        "I absolutely love this product! It's amazing!",
        "Outstanding quality and fantastic service!",
        "Best purchase I've ever made!",
        "Incredible! Will definitely buy again!",
        "Perfect! Exceeded all expectations!",
        "Wonderful experience from start to finish!",
        "Highly recommend to everyone!",
        "Exceptional quality and great value!",
    ] * multiplier

    negative_texts = [
        "Worst product ever. Complete waste of money.",
        "Terrible quality and horrible service.",
        "Very disappointed. Would not recommend.",
        "Poor quality. Broke after one use.",
        "Awful experience. Save your money!",
        "Completely unsatisfied with this purchase.",
        "Not worth a single penny.",
        "Regret buying this product.",
    ] * multiplier

    neutral_texts = [
        "It's okay, nothing special.",
        "Average product, met basic expectations.",
        "Decent quality, works as described.",
        "Fair product for the price.",
        "Acceptable, nothing to complain about.",
        "Standard quality, nothing impressive.",
        "Mediocre, but functional.",
        "Neither good nor bad, just average.",
    ] * multiplier

    texts = positive_texts + negative_texts + neutral_texts
    labels = ([1] * len(positive_texts) +
              [0] * len(negative_texts) +
              [2] * len(neutral_texts))

    df = pd.DataFrame({'text': texts, 'sentiment': labels})
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


@st.cache_resource
def train_pipeline(size, model_type):
    """Train the full pipeline and cache the results."""
    df = create_demo_dataset(size)

    preprocessor = TextPreprocessor(remove_stopwords=True, lemmatize=True)
    df = preprocessor.preprocess_dataframe(df)

    X_train, X_test, y_train, y_test, feature_extractor = prepare_train_test_split(df)

    results = {}

    if model_type in ('all', 'lr'):
        lr = SentimentModelTrainer('logistic_regression')
        lr.train(X_train, y_train)
        eval_result = lr.evaluate(X_test, y_test)
        results['Logistic Regression'] = {**eval_result, 'trainer': lr}

    if model_type in ('all', 'rf'):
        rf = SentimentModelTrainer('random_forest')
        rf.train(X_train, y_train)
        eval_result = rf.evaluate(X_test, y_test)
        results['Random Forest'] = {**eval_result, 'trainer': rf}

    best_name = max(results, key=lambda k: results[k]['accuracy'])

    return {
        'results': results,
        'best_name': best_name,
        'best_trainer': results[best_name]['trainer'],
        'preprocessor': preprocessor,
        'feature_extractor': feature_extractor,
        'y_test': y_test,
        'dataset_size': len(df),
    }


# ── Page config ──────────────────────────────────────────────
st.set_page_config(page_title="Sentiment Analysis", page_icon="📊", layout="wide")
st.title("📊 Social Media Sentiment Analysis")
st.markdown("Train ML models on demo data and predict sentiment on custom text.")

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")
    size = st.selectbox("Dataset size", ['small', 'medium', 'large'], index=1)
    model_choice = st.selectbox("Model", ['all', 'lr', 'rf'], index=0,
                                format_func=lambda x: {
                                    'all': 'All models',
                                    'lr': 'Logistic Regression',
                                    'rf': 'Random Forest',
                                }[x])
    train_btn = st.button("Train Models", type="primary", use_container_width=True)

# ── Train / load cached pipeline ─────────────────────────────
if train_btn or 'pipeline' in st.session_state:
    if train_btn:
        with st.spinner("Training models..."):
            st.session_state.pipeline = train_pipeline(size, model_choice)

    pipe = st.session_state.pipeline

    # ── Metrics row ──────────────────────────────────────────
    st.subheader("Model Performance")
    cols = st.columns(len(pipe['results']))
    for col, (name, res) in zip(cols, pipe['results'].items()):
        trophy = " 🏆" if name == pipe['best_name'] and len(pipe['results']) > 1 else ""
        col.metric(f"{name}{trophy}", f"{res['accuracy']:.4f}")

    # ── Confusion matrices ───────────────────────────────────
    st.subheader("Confusion Matrices")
    cm_cols = st.columns(len(pipe['results']))
    class_names = ['Negative', 'Positive', 'Neutral']

    for col, (name, res) in zip(cm_cols, pipe['results'].items()):
        cm = confusion_matrix(pipe['y_test'], res['predictions'])
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        col.markdown(f"**{name}**")
        col.dataframe(cm_df, use_container_width=True)

    # ── Interactive prediction ───────────────────────────────
    st.subheader("Try It Out")
    user_text = st.text_area("Enter text to analyze", placeholder="Type a review or social media post...")

    if user_text:
        processed = pipe['preprocessor'].preprocess(user_text)
        features = pipe['feature_extractor'].transform([processed])
        pred = pipe['best_trainer'].predict(features)[0]
        emoji = SENTIMENT_EMOJI[pred]
        label = SENTIMENT_MAP[pred]
        st.success(f"**Predicted sentiment:** {label} {emoji}")

    # ── Dataset info ─────────────────────────────────────────
    with st.expander("Dataset Info"):
        st.write(f"Samples: **{pipe['dataset_size']}**")
        df_preview = create_demo_dataset(size)
        dist = df_preview['sentiment'].value_counts().rename(index=SENTIMENT_MAP)
        st.bar_chart(dist)
else:
    st.info("Click **Train Models** in the sidebar to get started.")
