import logging

from src import config
from src.evaluate import evaluate_model, find_best_threshold
from src.train import main as train_main

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_pipeline():
    logger.info("STEP 1-3: Preprocessing, training, and hyperparameter tuning...")
    best_name, best_model, X_test, y_test, mlflow_run_id = train_main()

    logger.info("STEP 4: Evaluating best model ('%s') on the test set...", best_name)
    evaluate_model(best_model, X_test, y_test, model_name=best_name, mlflow_run_id=mlflow_run_id)

    business_threshold = find_best_threshold(best_model, X_test, y_test, min_precision=0.9)
    logger.info("Recommended decision threshold for production: %.4f", business_threshold)

    logger.info("Pipeline complete. Model saved at: %s", config.BEST_MODEL_PATH)
    logger.info("Run 'mlflow ui' in the project directory to explore all experiments.")


if __name__ == "__main__":
    run_pipeline()
