import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Imports from your project modules
from xai_evaluation_framework.data_loader import load_pima
from xai_evaluation_framework.data_preprocessing import preprocess_pima, preprocess_data
from xai_evaluation_framework.model_training import train_random_forest
from xai_evaluation_framework.explainers import generate_explanations
from xai_evaluation_framework.evaluation.evaluator import XAIEvaluator
from xai_evaluation_framework.evaluation.stability_metrics import add_noise

# --- Dashboard Configuration ---
st.set_page_config(
    page_title="Medical XAI Evaluation Dashboard",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 XAI Evaluation Framework: Diabetes Diagnosis")
st.markdown("Evaluating model reliability and explanation trustworthiness using the PIMA dataset.")


# --- Backend Logic (Cached for Performance) ---
@st.cache_resource
def run_evaluation_pipeline():
    # 1. Load & Process
    df = load_pima()
    df_proc = preprocess_pima(df)
    X_train, X_test, y_train, y_test = preprocess_data(df_proc, target="Outcome")

    # 2. Train (Retrieves updated metrics dictionary: Acc, Prec, Rec, F1)
    model, medical_metrics = train_random_forest(X_train, y_train, X_test, y_test)

    # 3. Generate XAI Explanations
    feature_names = df_proc.drop(columns=["Outcome"]).columns.tolist()
    X_sample = X_test[:5]
    lime_explanations = generate_explanations(model, X_train, X_sample, feature_names)

    # 4. Perturbation for Stability Analysis
    original_pred = model.predict_proba(X_sample)[:, 1]
    X_perturbed = add_noise(X_sample, noise_level=0.01)
    perturbed_pred = model.predict_proba(X_perturbed)[:, 1]

    # 5. XAI Framework Evaluation
    evaluator = XAIEvaluator()
    xai_results = evaluator.evaluate_lime_only(
        lime_explanations=lime_explanations,
        original_pred=original_pred,
        perturbed_pred=perturbed_pred
    )

    return medical_metrics, xai_results, X_sample, original_pred


# Execute the pipeline
metrics, xai_eval, samples, predictions = run_evaluation_pipeline()

# --- SECTION 1: MEDICAL PERFORMANCE METRICS ---
st.header("1. Model Diagnostic Performance")
st.info("Medical diagnosis requires high Recall (finding all sick patients) and Precision (diagnosis reliability).")

m1, m2, m3, m4 = st.columns(4)
# Accessing keys from the metrics dictionary returned by model_training.py
m1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
m2.metric("Precision", f"{metrics['precision']:.2%}", help="How many predicted diabetic cases were actually diabetic?")
m3.metric("Recall", f"{metrics['recall']:.2%}", help="How many actual diabetic cases did the model find?")
m4.metric("F1-Score", f"{metrics['f1']:.2%}", help="Balanced harmonic mean of Precision and Recall.")

st.markdown("---")

# --- SECTION 2: XAI EVALUATION RESULTS ---
st.header("2. XAI Trustworthiness Metrics")
x1, x2, x3 = st.columns(3)

with x1:
    st.subheader("Quality")
    st.write(f"**Result:** {xai_eval['quality']}")
    st.caption("Validates if the explanation follows the model's internal logic.")

with x2:
    st.subheader("Stability")
    st.write(f"**Score:** {xai_eval['stability']:.4f}")
    st.caption("Measures if small data changes cause erratic explanation shifts.")

with x3:
    st.subheader("Usability")
    st.write(f"**Level:** {xai_eval['usability']}")
    st.caption("Heuristic check for human interpretability based on feature count.")

# --- SECTION 3: STABILITY VISUALIZATION ---
st.subheader("Visual Stability Analysis")
fig, ax = plt.subplots(figsize=(8, 4))
# FIX: Store the bar object to add numeric labels
bars = ax.bar(["LIME Consistency"], [xai_eval['stability']], color='#FF4B4B', width=0.4)
ax.bar_label(bars, padding=3, fmt='%.4f', fontweight='bold')

# FIX: Dynamic Y-axis limit to ensure small scores (e.g., 0.016) are visible
ax.set_ylim(0, max(0.05, xai_eval['stability'] * 2))
ax.set_ylabel("Perturbation Variance (Lower is Better)")
ax.grid(axis='y', linestyle='--', alpha=0.6)
st.pyplot(fig)

st.markdown("---")

# --- SECTION 4: ANALYZED DATA INSTANCES ---
st.header("3. Patient Sample Data")
st.write("Below are the specific patient records analyzed by the XAI engine.")
display_df = samples.copy()
display_df['Model Probability'] = predictions
st.dataframe(
    display_df.style.background_gradient(subset=['Model Probability'], cmap='YlOrRd')
    .format({'Model Probability': '{:.2%}'})
)

st.success("Dashboard successfully loaded: Analysis complete for PIMA Diabetes Dataset.")