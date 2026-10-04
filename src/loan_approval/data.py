from pathlib import Path

import pandas as pd


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset was not found: {path}")

    dataframe = pd.read_csv(path)

    if dataframe.empty:
        raise ValueError(f"Dataset is empty: {path}")

    return dataframe


def split_features_and_target(
    dataframe: pd.DataFrame,
    target_column: str,
) -> tuple[pd.DataFrame, pd.Series]:
    if target_column not in dataframe.columns:
        available_columns = dataframe.columns.tolist()
        raise ValueError(
            f"Target column '{target_column}' is absent. Available columns: {available_columns}"
        )

    features = dataframe.drop(columns=[target_column])
    target = dataframe[target_column]
    return features, target
