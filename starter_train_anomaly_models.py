"""
train_anomaly_models.py

Trains two unsupervised anomaly detectors -- Isolation Forest and
One-Class SVM -- on NORMAL-ONLY engineered features, then evaluates both
against labeled semi-synthetic anomaly data (data/labeled/labeled_features.csv,
which must include a ground-truth `label` column with values like
"normal", "cpu_spike", "cryptomining", "resource_abuse", "dos").

Also trains a lightweight second-stage classifier that, given a window
already flagged as anomalous, predicts WHICH type of anomaly it is.

Run:
    python train_anomaly_models.py
"""
import os

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

NORMAL_FEATURES_PATH = "data/processed/features.csv"
LABELED_FEATURES_PATH = "data/labeled/labeled_features.csv"
MODEL_DIR = "models/saved"
os.makedirs(MODEL_DIR, exist_ok=True)

CONTAMINATION = 0.05  # rough estimate of expected anomaly rate; tune on your data


def get_feature_cols(df: pd.DataFrame) -> list:
    exclude = {"timestamp", "vm_id", "label"}
    return [c for c in df.columns if c not in exclude]


def train_detectors():
    normal_df = pd.read_csv(NORMAL_FEATURES_PATH)
    feature_cols = get_feature_cols(normal_df)

    scaler = StandardScaler().fit(normal_df[feature_cols])
    X_normal = scaler.transform(normal_df[feature_cols])

    iso_forest = IsolationForest(
        n_estimators=200, contamination=CONTAMINATION, random_state=42
    ).fit(X_normal)

    ocsvm = OneClassSVM(kernel="rbf", nu=CONTAMINATION, gamma="scale").fit(X_normal)

    joblib.dump(scaler, os.path.join(MODEL_DIR, "anomaly_scaler.pkl"))
    joblib.dump(iso_forest, os.path.join(MODEL_DIR, "isolation_forest.pkl"))
    joblib.dump(ocsvm, os.path.join(MODEL_DIR, "one_class_svm.pkl"))
    print("Trained and saved Isolation Forest + One-Class SVM.")

    return scaler, iso_forest, ocsvm, feature_cols


def evaluate_detectors(scaler, iso_forest, ocsvm, feature_cols):
    if not os.path.exists(LABELED_FEATURES_PATH):
        print(f"\nNo labeled eval set found at {LABELED_FEATURES_PATH} -- skipping evaluation.")
        return None

    labeled_df = pd.read_csv(LABELED_FEATURES_PATH)
    X = scaler.transform(labeled_df[feature_cols])
    y_true = (labeled_df["label"] != "normal").astype(int)

    # -1 = anomaly, 1 = normal for both sklearn detectors -> convert to 1 = anomaly
    iso_pred = (iso_forest.predict(X) == -1).astype(int)
    ocsvm_pred = (ocsvm.predict(X) == -1).astype(int)
    ensemble_pred = ((iso_pred + ocsvm_pred) >= 1).astype(int)  # flag if either fires

    for name, pred in [("IsolationForest", iso_pred), ("OneClassSVM", ocsvm_pred), ("Ensemble(OR)", ensemble_pred)]:
        print(f"\n=== {name} ===")
        print(classification_report(y_true, pred, target_names=["normal", "anomaly"]))
        try:
            print(f"ROC-AUC: {roc_auc_score(y_true, pred):.3f}")
        except ValueError:
            pass

    return labeled_df, iso_pred, ocsvm_pred


def train_anomaly_type_classifier(labeled_df: pd.DataFrame, feature_cols: list):
    """Second stage: given a flagged window, predict WHICH anomaly type it is."""
    anomalies = labeled_df[labeled_df["label"] != "normal"].copy()
    if anomalies["label"].nunique() < 2:
        print("\nNot enough anomaly-type diversity to train the type classifier yet.")
        return

    X = anomalies[feature_cols]
    y = anomalies["label"]

    clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    clf.fit(X, y)
    joblib.dump(clf, os.path.join(MODEL_DIR, "anomaly_type_classifier.pkl"))
    print("\nTrained and saved second-stage anomaly-type classifier.")
    print("Classes:", list(clf.classes_))


if __name__ == "__main__":
    scaler, iso_forest, ocsvm, feature_cols = train_detectors()
    result = evaluate_detectors(scaler, iso_forest, ocsvm, feature_cols)
    if result is not None:
        labeled_df, _, _ = result
        train_anomaly_type_classifier(labeled_df, feature_cols)
