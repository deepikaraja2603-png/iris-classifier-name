"""
app.py
------
A simple AI-powered web app (built with Streamlit) that lets a user enter
flower measurements and get a live species prediction from the model
trained in train_model.py (originally built in Module 4).

Run with:
    streamlit run app.py
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Iris Flower Classifier", page_icon="🌸", layout="centered")


@st.cache_resource
def load_artifacts():
    model = joblib.load("model/iris_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    target_names = joblib.load("model/target_names.pkl")
    return model, scaler, target_names


model, scaler, target_names = load_artifacts()

st.title("🌸 Iris Flower Classifier")
st.write(
    "An AI-powered mini app that predicts the species of an iris flower "
    "from its measurements, using a K-Nearest Neighbors model trained in "
    "**Module 4: Machine Learning Fundamentals**."
)

st.divider()
st.subheader("Enter flower measurements")

col1, col2 = st.columns(2)
with col1:
    sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.8, 0.1)
    petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 4.0, 0.1)
with col2:
    sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.0, 0.1)
    petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 1.2, 0.1)

if st.button("Predict species", type="primary"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]

    st.success(f"Predicted species: **{target_names[prediction]}**")

    prob_df = pd.DataFrame({
        "Species": target_names,
        "Confidence": probabilities
    }).sort_values("Confidence", ascending=False)

    st.write("Confidence per species:")
    st.bar_chart(prob_df.set_index("Species"))

st.divider()
with st.expander("How this works"):
    st.write(
        "1. Your inputs are scaled the same way the training data was scaled.\n"
        "2. The K-Nearest Neighbors model looks at the 5 most similar flowers "
        "it saw during training.\n"
        "3. It predicts the species that shows up most often among those neighbors, "
        "and reports how confident it is in each possible species."
    )

st.caption("Built as part of Module 5: AI Tools & Mini Project — extending the Module 4 ML model into an interactive app.")
