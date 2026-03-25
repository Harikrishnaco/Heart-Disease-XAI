from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

def train_random_forest(X_train, y_train, X_test, y_test):
    """
    Trains a Random Forest classifier and returns the model along with a
    dictionary of performance metrics relevant to medical diagnosis.
    """
    # Initialize the model with a fixed random state for reproducibility
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    # Fit the model on the training data
    model.fit(X_train, y_train)

    # Generate predictions on the test set
    y_pred = model.predict(X_test)

    # 1. Detailed console output for debugging and clinical review
    print("\n--- Detailed Medical Classification Report ---")
    # target_names maps 0 to Healthy and 1 to Diabetic based on the PIMA dataset structure
    print(classification_report(y_test, y_pred, target_names=["Healthy", "Diabetic"]))

    # 2. Package metrics into a dictionary for the UI (showcase_app.py)
    # and experimental tracking (run_experiments.py)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred)
    }

    return model, metrics