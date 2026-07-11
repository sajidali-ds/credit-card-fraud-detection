import json
import logging

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def evaluate_model(model, X_test, y_test, model_name="model", mlflow_run_id=None):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, target_names=["genuine", "fraud"], output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    logger.info("Classification report:\n%s",
                 classification_report(y_test, y_pred, target_names=["genuine", "fraud"]))
    logger.info("Confusion matrix:\n%s", cm)
    logger.info("ROC-AUC: %.4f | PR-AUC (Average Precision): %.4f", roc_auc, pr_auc)

    metrics = {
        "model_name": model_name,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    with open(config.METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved to %s", config.METRICS_PATH)

    _plot_curves(y_test, y_proba, roc_auc, pr_auc)

    if mlflow_run_id is not None:
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
        with mlflow.start_run(run_name="evaluation", nested=False):
            mlflow.set_tag("stage", "test_set_evaluation")
            mlflow.set_tag("parent_training_run_id", mlflow_run_id)
            mlflow.set_tag("evaluated_model", model_name)
            mlflow.log_metric("test_roc_auc", roc_auc)
            mlflow.log_metric("test_pr_auc", pr_auc)
            mlflow.log_metric("test_fraud_precision", report["fraud"]["precision"])
            mlflow.log_metric("test_fraud_recall", report["fraud"]["recall"])
            mlflow.log_metric("test_fraud_f1", report["fraud"]["f1-score"])
            mlflow.log_artifact(config.METRICS_PATH)
            mlflow.log_artifact(f"{config.OUTPUTS_DIR}/roc_pr_curves.png")
        logger.info("Logged evaluation metrics to MLflow (linked to run %s)", mlflow_run_id)

    return metrics


def _plot_curves(y_test, y_proba, roc_auc, pr_auc):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_proba)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend()

    axes[1].plot(recall, precision, label=f"PR curve (AP = {pr_auc:.4f})", color="darkorange")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend()

    plt.tight_layout()
    out_path = f"{config.OUTPUTS_DIR}/roc_pr_curves.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    logger.info("Saved ROC/PR curve plot to %s", out_path)


def find_best_threshold(model, X_test, y_test, min_precision=0.9):
    """
    By default sklearn uses a 0.5 probability threshold, which is rarely
    optimal for fraud detection. This helper finds the threshold that
    maximises recall while keeping precision at or above `min_precision` -
    a realistic business constraint (limit false alarms, catch more fraud).
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

    # precision/recall arrays are 1 longer than thresholds
    valid = precision[:-1] >= min_precision
    if not valid.any():
        logger.warning("No threshold reaches precision >= %.2f; returning default 0.5", min_precision)
        return 0.5

    best_idx = np.argmax(recall[:-1][valid])
    best_threshold = thresholds[valid][best_idx]
    logger.info(
        "Best threshold for precision >= %.2f: %.4f (recall = %.4f)",
        min_precision, best_threshold, recall[:-1][valid][best_idx],
    )
    return best_threshold


if __name__ == "__main__":
    from src.data_preprocessing import get_train_test_data

    model = joblib.load(config.BEST_MODEL_PATH)
    _, X_test, _, y_test = get_train_test_data()
    evaluate_model(model, X_test, y_test, model_name="best_model")
