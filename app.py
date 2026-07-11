import json
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src import config

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
)
st.write("Current Working Directory:", os.getcwd())
st.write("BASE_DIR:", config.BASE_DIR)
st.write("MODELS_DIR:", config.MODELS_DIR)
st.write("MODELS_DIR exists:", os.path.exists(config.MODELS_DIR))

if os.path.exists(config.MODELS_DIR):
    st.write("Models folder:", os.listdir(config.MODELS_DIR))

st.write("BEST_MODEL_PATH:", config.BEST_MODEL_PATH)
st.write("Model exists:", os.path.exists(config.BEST_MODEL_PATH))
st.write("Scaler exists:", os.path.exists(config.SCALER_PATH))

@st.cache_resource
def load_model_and_scaler():
    try:
        model = joblib.load(config.BEST_MODEL_PATH)
        scaler = joblib.load(config.SCALER_PATH)
        return model, scaler
    except Exception as e:
        st.exception(e)
        return None, None


def score_transactions(model, scaler, df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    X = df.copy()
    cols_to_scale = [c for c in ["Time", "Amount"] if c in X.columns]
    X_scaled = X.copy()
    X_scaled[cols_to_scale] = scaler.transform(X[cols_to_scale])

    proba = model.predict_proba(X_scaled)[:, 1]
    result = df.copy()
    result["fraud_probability"] = proba
    result["is_fraud"] = (proba >= threshold).astype(int)
    return result



st.sidebar.title("💳 Fraud Detection")
st.sidebar.markdown("An industry-style ML pipeline for detecting fraudulent transactions.")

model, scaler = load_model_and_scaler()

if model is None:
    st.sidebar.error("No trained model found.")
    st.warning(
        "No trained model found at `models/best_fraud_model.joblib`.\n\n"
        "Run the training pipeline first:\n\n"
        "```bash\npython main.py\n```"
    )
    st.stop()

st.sidebar.success("Model loaded ✅")
st.sidebar.caption(f"Model type: `{type(model.named_steps['clf']).__name__}`")

threshold = st.sidebar.slider(
    "Fraud decision threshold",
    min_value=0.0, max_value=1.0, value=0.5, step=0.01,
    help="A transaction is flagged as fraud if predicted probability ≥ this value. "
         "Lower it to catch more fraud (more false alarms); raise it to reduce false alarms (may miss fraud).",
)

page = st.sidebar.radio("Navigate", ["Batch Prediction", "Single Transaction", "Model Performance"])

st.sidebar.markdown("---")
st.sidebar.caption("Track experiments with MLflow:")
st.sidebar.code("mlflow ui", language="bash")


if page == "Batch Prediction":
    st.title("Batch Transaction Scoring")
    st.write(
        "Upload a CSV with the same columns as the training data "
        "(`Time`, `V1`...`V28`, `Amount` — `Class` optional)."
    )

    uploaded = st.file_uploader("Upload transactions CSV", type=["csv"])

    if uploaded is not None:
        raw_df = pd.read_csv(uploaded)
        has_labels = "Class" in raw_df.columns
        feature_df = raw_df.drop(columns=["Class"]) if has_labels else raw_df

        with st.spinner("Scoring transactions..."):
            scored = score_transactions(model, scaler, feature_df, threshold)
            if has_labels:
                scored["actual_class"] = raw_df["Class"].values

        n_fraud = int(scored["is_fraud"].sum())
        col1, col2, col3 = st.columns(3)
        col1.metric("Transactions scanned", f"{len(scored):,}")
        col2.metric("Flagged as fraud", f"{n_fraud:,}")
        col3.metric("Flag rate", f"{100 * n_fraud / len(scored):.3f}%")

        st.subheader("Flagged transactions")
        flagged = scored[scored["is_fraud"] == 1].sort_values("fraud_probability", ascending=False)
        st.dataframe(flagged, use_container_width=True)

        st.subheader("Fraud probability distribution")
        fig = px.histogram(
            scored, x="fraud_probability", nbins=50, log_y=True,
            color="is_fraud", color_discrete_map={0: "#4C78A8", 1: "#E45756"},
            labels={"fraud_probability": "Predicted fraud probability", "is_fraud": "Flagged"},
        )
        fig.add_vline(x=threshold, line_dash="dash", line_color="black",
                       annotation_text=f"threshold = {threshold}")
        st.plotly_chart(fig, use_container_width=True)

        if has_labels:
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(scored["actual_class"], scored["is_fraud"])
            st.subheader("Confusion matrix (since this file had true labels)")
            cm_fig = px.imshow(
                cm, text_auto=True, color_continuous_scale="Blues",
                x=["Predicted genuine", "Predicted fraud"],
                y=["Actual genuine", "Actual fraud"],
            )
            st.plotly_chart(cm_fig, use_container_width=True)

        st.download_button(
            "Download scored results as CSV",
            data=scored.to_csv(index=False).encode("utf-8"),
            file_name="scored_transactions.csv",
            mime="text/csv",
        )
    else:
        st.info("Waiting for a CSV upload. Don't have one handy? Try the 'Single Transaction' tab instead.")


elif page == "Single Transaction":
    st.title("Check a Single Transaction")
    st.write("Manually enter transaction details (or paste values from a real row) to get an instant fraud score.")

    with st.form("single_txn_form"):
        col1, col2 = st.columns(2)
        with col1:
            time_val = st.number_input("Time (seconds since first transaction)", min_value=0.0, value=0.0)
            amount_val = st.number_input("Amount", min_value=0.0, value=100.0)
        with col2:
            st.caption("V1–V28 are anonymised PCA features from the original dataset. Defaulting to 0 (average) — adjust if you have real values.")

        v_values = {}
        v_cols = st.columns(4)
        for i in range(1, 29):
            with v_cols[(i - 1) % 4]:
                v_values[f"V{i}"] = st.number_input(f"V{i}", value=0.0, key=f"v_{i}", step=0.1)

        submitted = st.form_submit_button("Check transaction")

    if submitted:
        row = {"Time": time_val, **v_values, "Amount": amount_val}
        single_df = pd.DataFrame([row])
        scored = score_transactions(model, scaler, single_df, threshold)

        proba = scored["fraud_probability"].iloc[0]
        is_fraud = scored["is_fraud"].iloc[0]

        if is_fraud:
            st.error(f"⚠️ Likely FRAUD — predicted probability: {proba:.4%}")
        else:
            st.success(f"✅ Looks genuine — predicted probability: {proba:.4%}")

        st.progress(min(float(proba), 1.0))


elif page == "Model Performance":
    st.title("Model Performance Dashboard")
    st.write("Metrics from the most recent evaluation on the held-out test set.")

    if not os.path.exists(config.METRICS_PATH):
        st.warning("No metrics found yet. Run `python main.py` to train and evaluate a model first.")
        st.stop()

    with open(config.METRICS_PATH) as f:
        metrics = json.load(f)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model", metrics["model_name"])
    col2.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
    col3.metric("PR-AUC (Avg. Precision)", f"{metrics['pr_auc']:.4f}")
    col4.metric("Fraud F1-score", f"{metrics['classification_report']['fraud']['f1-score']:.4f}")

    st.subheader("Classification report")
    report_df = pd.DataFrame(metrics["classification_report"]).T
    st.dataframe(report_df, use_container_width=True)

    st.subheader("Confusion matrix")
    cm = np.array(metrics["confusion_matrix"])
    cm_fig = px.imshow(
        cm, text_auto=True, color_continuous_scale="Blues",
        x=["Predicted genuine", "Predicted fraud"],
        y=["Actual genuine", "Actual fraud"],
    )
    st.plotly_chart(cm_fig, use_container_width=True)

    curves_path = f"{config.OUTPUTS_DIR}/roc_pr_curves.png"
    if os.path.exists(curves_path):
        st.subheader("ROC & Precision-Recall Curves")
        st.image(curves_path, use_container_width=True)

    st.info(
        "For full experiment history (every model tried, every hyperparameter "
        "combination, and this run's lineage), launch the MLflow UI:\n\n"
        "```bash\nmlflow ui\n```"
    )
