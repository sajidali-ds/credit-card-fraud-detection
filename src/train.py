import logging
import time

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from xgboost import XGBClassifier

from src import config
from src.data_preprocessing import get_train_test_data

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def setup_mlflow():
    """Point MLflow at the local (or remote) tracking server and experiment."""
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_NAME)
    logger.info(
        "MLflow tracking URI: %s | experiment: %s",
        config.MLFLOW_TRACKING_URI, config.MLFLOW_EXPERIMENT_NAME,
    )


def build_search_spaces():
    """
    Defines each model + its hyperparameter search space.
    Returns a dict: name -> (pipeline, param_distributions)
    """
    cv_strategy = StratifiedKFold(
        n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE
    )

    searches = {}

    # ---------------- Logistic Regression (fast, interpretable baseline) ----
    lr_pipeline = ImbPipeline([
        ("smote", SMOTE(random_state=config.RANDOM_STATE)),
        ("clf", LogisticRegression(max_iter=2000, random_state=config.RANDOM_STATE)),
    ])
    lr_params = {
        "clf__C": np.logspace(-3, 2, 20),
        "clf__class_weight": [None, "balanced"],
    }
    searches["logistic_regression"] = (lr_pipeline, lr_params)

    # ---------------- Random Forest (strong non-linear baseline) -----------
    rf_pipeline = ImbPipeline([
        ("smote", SMOTE(random_state=config.RANDOM_STATE)),
        ("clf", RandomForestClassifier(random_state=config.RANDOM_STATE, n_jobs=-1)),
    ])
    rf_params = {
        "clf__n_estimators": [100, 200, 300, 400],
        "clf__max_depth": [4, 6, 8, 10, None],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf": [1, 2, 4],
        "clf__class_weight": [None, "balanced", "balanced_subsample"],
    }
    searches["random_forest"] = (rf_pipeline, rf_params)

    # ---------------- XGBoost (industry standard for tabular fraud data) ---
    xgb_pipeline = ImbPipeline([
        ("smote", SMOTE(random_state=config.RANDOM_STATE)),
        ("clf", XGBClassifier(
            eval_metric="aucpr",
            random_state=config.RANDOM_STATE,
            n_jobs=-1,
            tree_method="hist",
        )),
    ])
    xgb_params = {
        "clf__n_estimators": [150, 250, 350],
        "clf__max_depth": [3, 4, 5, 6],
        "clf__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "clf__subsample": [0.7, 0.8, 1.0],
        "clf__colsample_bytree": [0.7, 0.8, 1.0],
        "clf__scale_pos_weight": [1, 5, 10],  # extra lever for imbalance
    }
    searches["xgboost"] = (xgb_pipeline, xgb_params)

    return searches, cv_strategy


def run_hyperparameter_search(X_train, y_train):
    """
    Runs RandomizedSearchCV for every candidate model and returns a dict of
    fitted RandomizedSearchCV objects, keyed by model name.
    """
    searches, cv_strategy = build_search_spaces()
    fitted = {}

    for name, (pipeline, param_dist) in searches.items():
        logger.info("=== Tuning %s ===", name)
        start = time.time()

        with mlflow.start_run(run_name=name, nested=True):
            search = RandomizedSearchCV(
                estimator=pipeline,
                param_distributions=param_dist,
                n_iter=config.N_ITER_SEARCH,
                scoring=config.TUNING_SCORING,
                cv=cv_strategy,
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
                verbose=1,
                refit=True,
            )
            search.fit(X_train, y_train)

            elapsed = time.time() - start
            logger.info(
                "%s -> best CV %s: %.4f (%.1fs) | best params: %s",
                name, config.TUNING_SCORING, search.best_score_, elapsed, search.best_params_,
            )

            # ---- MLflow logging: params, CV metric, search metadata ----
            mlflow.set_tag("model_family", name)
            mlflow.set_tag("stage", "hyperparameter_search")
            mlflow.log_param("n_iter_search", config.N_ITER_SEARCH)
            mlflow.log_param("cv_folds", config.CV_FOLDS)
            mlflow.log_params({f"best_{k}": v for k, v in search.best_params_.items()})
            mlflow.log_metric(f"cv_{config.TUNING_SCORING}", search.best_score_)
            mlflow.log_metric("search_time_seconds", elapsed)

        fitted[name] = search

    return fitted


def select_best_model(fitted_searches: dict):
    """Picks the model with the highest cross-validated score."""
    best_name = max(fitted_searches, key=lambda n: fitted_searches[n].best_score_)
    best_search = fitted_searches[best_name]
    logger.info(
        "Best model overall: %s (CV %s = %.4f)",
        best_name, config.TUNING_SCORING, best_search.best_score_,
    )
    return best_name, best_search


def main():
    setup_mlflow()

    X_train, X_test, y_train, y_test = get_train_test_data()

    with mlflow.start_run(run_name="fraud_detection_experiment") as parent_run:
        mlflow.set_tag("dataset_rows", len(X_train) + len(X_test))
        mlflow.log_param("test_size", config.TEST_SIZE)
        mlflow.log_param("tuning_scoring", config.TUNING_SCORING)

        fitted_searches = run_hyperparameter_search(X_train, y_train)
        best_name, best_search = select_best_model(fitted_searches)

        best_model = best_search.best_estimator_
        joblib.dump(best_model, config.BEST_MODEL_PATH)
        logger.info("Saved best model ('%s') to %s", best_name, config.BEST_MODEL_PATH)

        # ---- Log + register the winning model in the parent run ----
        mlflow.set_tag("best_model", best_name)
        mlflow.log_metric(f"best_cv_{config.TUNING_SCORING}", best_search.best_score_)

        model_info = mlflow.sklearn.log_model(
            sk_model=best_model,
            name="model",
            registered_model_name=config.MLFLOW_REGISTERED_MODEL_NAME,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        )
        logger.info(
            "Registered model '%s' (run_id=%s) in MLflow Model Registry",
            config.MLFLOW_REGISTERED_MODEL_NAME, parent_run.info.run_id,
        )
        logger.info("View this run: mlflow ui --backend-store-uri %s", config.MLFLOW_TRACKING_URI)

    # Persist test split to disk-free objects for the evaluate step by returning them
    return best_name, best_model, X_test, y_test, parent_run.info.run_id


if __name__ == "__main__":
    main()
