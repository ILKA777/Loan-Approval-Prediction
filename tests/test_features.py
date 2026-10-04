import pandas as pd
import pytest

from loan_approval.features import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    build_preprocessor,
    select_model_features,
)


def create_features_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2],
            "person_age": [25, 40],
            "person_income": [50_000, 70_000],
            "person_home_ownership": ["RENT", "MORTGAGE"],
            "person_emp_length": [2.0, None],
            "loan_intent": ["PERSONAL", "EDUCATION"],
            "loan_grade": ["B", "A"],
            "loan_amnt": [10_000, 15_000],
            "loan_int_rate": [12.5, 8.5],
            "loan_percent_income": [0.2, 0.21],
            "cb_person_default_on_file": ["N", "Y"],
            "cb_person_cred_hist_length": [4, 10],
        }
    )


def test_select_model_features_returns_only_declared_features() -> None:
    dataframe = create_features_dataframe()

    result = select_model_features(dataframe)

    expected_columns = [*NUMERICAL_FEATURES, *CATEGORICAL_FEATURES]

    assert result.columns.tolist() == expected_columns
    assert "id" not in result.columns


def test_select_model_features_raises_for_missing_column() -> None:
    dataframe = create_features_dataframe().drop(columns=["loan_grade"])

    with pytest.raises(ValueError, match="loan_grade"):
        select_model_features(dataframe)


def test_preprocessor_transforms_all_rows() -> None:
    dataframe = create_features_dataframe()
    features = select_model_features(dataframe)

    transformed = build_preprocessor().fit_transform(features)

    assert transformed.shape[0] == len(dataframe)
    assert transformed.shape[1] > len(NUMERICAL_FEATURES)
