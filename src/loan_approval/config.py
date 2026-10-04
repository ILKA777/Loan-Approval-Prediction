from pathlib import Path
from typing import Final

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

TRAIN_FILE = PROJECT_ROOT / "train.csv"
TEST_FILE = PROJECT_ROOT / "test.csv"
SAMPLE_SUBMISSION_FILE = PROJECT_ROOT / "sample_submission.csv"

MODEL_FILE = MODELS_DIR / "loan_approval_model.joblib"
SUBMISSION_FILE = REPORTS_DIR / "submission.csv"

ID_COLUMN: Final = "id"
TARGET_COLUMN: Final = "loan_status"

RANDOM_STATE: Final = 42
TEST_SIZE: Final = 0.2
