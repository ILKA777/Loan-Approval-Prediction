from pathlib import Path

import pandas as pd
import pytest

from loan_approval.data import load_csv, split_features_and_target


def test_split_features_and_target_returns_expected_data() -> None:
    dataframe = pd.DataFrame(
        {
            "person_age": [25, 41],
            "person_income": [50_000, 80_000],
            "loan_status": [0, 1],
        }
    )

    features, target = split_features_and_target(dataframe, "loan_status")

    assert features.columns.tolist() == ["person_age", "person_income"]
    assert target.tolist() == [0, 1]


def test_split_features_and_target_raises_for_missing_target() -> None:
    dataframe = pd.DataFrame({"person_age": [25, 41]})

    with pytest.raises(ValueError, match="loan_status"):
        split_features_and_target(dataframe, "loan_status")


def test_load_csv_raises_for_missing_file(tmp_path: Path) -> None:
    missing_file = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError, match="Dataset was not found"):
        load_csv(missing_file)
