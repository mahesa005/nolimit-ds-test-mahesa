import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import streamlit as st
from preprocess import preprocess
from predict import predict

st.set_page_config(page_title="Sentiment Analysis", layout="centered")

st.title("Sentiment Analysis")
st.caption("Model: cardiffnlp/twitter-roberta-base-sentiment")

text_input = st.text_area("Enter text to analyse", height=120, placeholder="Type something...")

if st.button("Analyse", use_container_width=True):
    if not text_input.strip():
        st.warning("Please enter some text.")
    else:
        cleaned = preprocess(text_input)
        result = predict(text_input)

        label = result["label"]
        score = result["score"]

        color = {"Positive": "green", "Neutral": "gray", "Negative": "red"}[label]

        st.markdown("---")
        col1, col2 = st.columns(2)
        col1.metric("Sentiment", label)
        col2.metric("Confidence", f"{score:.1%}")

        st.progress(score)

        with st.expander("Preprocessed text"):
            st.code(cleaned, language=None)
