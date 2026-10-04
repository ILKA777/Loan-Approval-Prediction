import logging

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from loan_approval.config import (
    MODEL_FILE,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
    TRAIN_FILE,
)
from loan_approval.data import load_csv, split_features_and_target
from loan_approval.features import build_preprocessor, select_model_features
from loan_approval.utils import configure_logging, ensure_directory

LOGGER = logging.getLogger(__name__)


def build_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1_000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def main() -> None:
    configure_logging()

    dataframe = load_csv(TRAIN_FILE)
    features, target = split_features_and_target(dataframe, TARGET_COLUMN)
    features = select_model_features(features)

    x_train, x_valid, y_train, y_valid = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    model = build_model()
    model.fit(x_train, y_train)

    validation_probabilities = model.predict_proba(x_valid)[:, 1]
    validation_roc_auc = roc_auc_score(y_valid, validation_probabilities)

    LOGGER.info("Validation ROC-AUC: %.5f", validation_roc_auc)

    ensure_directory(MODEL_FILE.parent)
    joblib.dump(model, MODEL_FILE)
    LOGGER.info("Model saved to %s", MODEL_FILE)


if __name__ == "__main__":
    main()
