# run_experiments.py

# -------- 1. Imports (relative, required for -m execution) --------
from .data_loader import load_pima
from .data_preprocessing import preprocess_pima, preprocess_data
from .model_training import train_random_forest
from .explainers import generate_explanations

from .evaluation.evaluator import XAIEvaluator
from .evaluation.stability_metrics import add_noise

# -------- 2. Load & Preprocess Dataset --------
# We use the PIMA Diabetes dataset for this medical XAI evaluation
df = load_pima()
df = preprocess_pima(df)

X_train, X_test, y_train, y_test = preprocess_data(df, target="Outcome")

# -------- 3. Train Model & Retrieve Medical Metrics --------
# The train_random_forest function now returns (model, metrics_dict)
model, metrics = train_random_forest(X_train, y_train, X_test, y_test)

# -------- 4. Generate XAI Explanations (LIME) --------
feature_names = df.drop(columns=["Outcome"]).columns.tolist()

# We analyze the first 5 samples to evaluate explanation stability
X_sample = X_test[:5]

lime_explanations = generate_explanations(
    model, X_train, X_sample, feature_names
)

# -------- 5. Perturbations (for Stability Evaluation) --------
# We check if a 1% change in data (noise) causes a massive change in model predictions
original_pred = model.predict_proba(X_sample)[:, 1]

X_perturbed = add_noise(X_sample, noise_level=0.01)
perturbed_pred = model.predict_proba(X_perturbed)[:, 1]

# -------- 6. Evaluate Framework Metrics --------
evaluator = XAIEvaluator()

results = evaluator.evaluate_lime_only(
    lime_explanations=lime_explanations,
    original_pred=original_pred,
    perturbed_pred=perturbed_pred
)

# -------- 7. Print Final Results --------
print("\n" + "="*40)
print(" MEDICAL MODEL PERFORMANCE SUMMARY ")
print("="*40)
# Accessing keys from the metrics dictionary returned in Step 3
print(f"Accuracy  : {metrics['accuracy']:.4f}")
print(f"Precision : {metrics['precision']:.4f} (Reliability of positive diagnosis)")
print(f"Recall    : {metrics['recall']:.4f} (Sensitivity to finding sick patients)")
print(f"F1-Score  : {metrics['f1']:.4f} (Balanced medical performance)")

print("\n" + "="*40)
print(" XAI EVALUATION FRAMEWORK RESULTS ")
print("="*40)
print(f"Quality Metrics   : {results['quality']}")
print(f"Stability Metrics : {results['stability']:.4f} (Lower is more consistent)")
print(f"Usability Metrics : {results['usability']} (Interpretability level)")
print("="*40)