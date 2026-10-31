"""Shared train/test stratified index generation."""
import pandas as pd
from sklearn.model_selection import train_test_split


def compute_global_indices(
    raw_df: pd.DataFrame,
    random_state: int = 42,
    test_size: float = 0.2,
) -> dict:
    """Return stratified train/test row indices from an already-loaded dataframe.

    Accepts raw_df (e.g. 7043 rows) and unpacks the two return values of
    train_test_split(raw_df.index, test_size=test_size, stratify=y, random_state=random_state)
    into train_idx and test_idx.
    """
    y = raw_df["Churn"]
    if not pd.api.types.is_numeric_dtype(y):
        y = y.map({"No": 0, "Yes": 1}).astype(int)

    train_idx, test_idx = train_test_split(
        raw_df.index,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )
    return {"train_idx": train_idx, "test_idx": test_idx}
