from typing import Final

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ID_COLUMN: Final = "id"

NUMERICAL_FEATURES: Final[tuple[str, ...]] = (
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
)

CATEGORICAL_FEATURES: Final[tuple[str, ...]] = (
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
)


def build_preprocessor() -> ColumnTransformer:
    numerical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numerical", numerical_pipeline, list(NUMERICAL_FEATURES)),
            ("categorical", categorical_pipeline, list(CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
    )


def select_model_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [*NUMERICAL_FEATURES, *CATEGORICAL_FEATURES]
    missing_columns = sorted(set(feature_columns) - set(dataframe.columns))

    if missing_columns:
        raise ValueError(f"Missing feature columns: {missing_columns}")

    return dataframe.loc[:, feature_columns].copy()
