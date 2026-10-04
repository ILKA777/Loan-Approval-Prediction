import logging

import joblib
import pandas as pd

from loan_approval.config import (
    ID_COLUMN,
    MODEL_FILE,
    SUBMISSION_FILE,
    TARGET_COLUMN,
    TEST_FILE,
)
from loan_approval.data import load_csv
from loan_approval.features import select_model_features
from loan_approval.utils import configure_logging, ensure_directory

LOGGER = logging.getLogger(__name__)


def create_submission(
    test_dataframe: pd.DataFrame,
    probabilities: list[float],
) -> pd.DataFrame:
    if ID_COLUMN not in test_dataframe.columns:
        raise ValueError(f"Missing ID column: {ID_COLUMN}")

    return pd.DataFrame(
        {
            ID_COLUMN: test_dataframe[ID_COLUMN],
            TARGET_COLUMN: probabilities,
        }
    )


def main() -> None:
    configure_logging()

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model was not found: {MODEL_FILE}. Run `poetry run python -m "
            "loan_approval.train` first."
        )

    test_dataframe = load_csv(TEST_FILE)
    test_features = select_model_features(test_dataframe)

    model = joblib.load(MODEL_FILE)
    probabilities = model.predict_proba(test_features)[:, 1].tolist()

    submission = create_submission(test_dataframe, probabilities)

    ensure_directory(SUBMISSION_FILE.parent)
    submission.to_csv(SUBMISSION_FILE, index=False)

    LOGGER.info("Submission saved to %s", SUBMISSION_FILE)


if __name__ == "__main__":
    main()
