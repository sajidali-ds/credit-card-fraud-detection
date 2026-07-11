import os


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_PATH = os.path.join(DATA_DIR, "creditcard.csv")

MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

BEST_MODEL_PATH = os.path.join(MODELS_DIR, "best_fraud_model.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
METRICS_PATH = os.path.join(OUTPUTS_DIR, "metrics.json")


RANDOM_STATE = 42

TEST_SIZE = 0.2


CV_FOLDS = 5


TUNING_SCORING = "average_precision"


N_ITER_SEARCH = 25


MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", f"sqlite:///{os.path.join(BASE_DIR, 'mlflow.db')}"
)
MLFLOW_EXPERIMENT_NAME = "credit-card-fraud-detection"
MLFLOW_REGISTERED_MODEL_NAME = "fraud-detection-model"

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
