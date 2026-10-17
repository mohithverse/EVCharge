"""Shared data preparation and evaluation helpers for tree-based regressors."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_FILE = Path(__file__).with_name("EVChargingStationUsage_cleaned.csv")
TARGET = "Energy (kWh)"
NUMERIC_FEATURES = ["Charging Time Minutes", "Hour"]
CATEGORICAL_FEATURES = ["Port Type", "Plug Type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_data() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load the cleaned dataset and make a reproducible train/test split."""
    data = pd.read_csv(DATA_FILE, usecols=FEATURES + [TARGET], low_memory=False)
    data = data.dropna(subset=[TARGET])
    return train_test_split(
        data[FEATURES],
        data[TARGET],
        test_size=0.20,
        random_state=42,
    )


def build_preprocessor() -> ColumnTransformer:
    """Impute missing values and one-hot encode categorical features."""
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                SimpleImputer(strategy="median"),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def report_results(
    model_name: str,
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """Print regression metrics and a few predictions."""
    predictions = np.maximum(model.predict(X_test), 0)
    print(f"Rows used: {len(y_test):,} test rows")
    print(f"\n{model_name} Results")
    print(f"MAE:  {mean_absolute_error(y_test, predictions):.3f} kWh")
    print(f"RMSE: {mean_squared_error(y_test, predictions) ** 0.5:.3f} kWh")
    print(f"R²:   {r2_score(y_test, predictions):.3f}")
    print("\nSample predictions:")
    results = pd.DataFrame(
        {"Actual Energy (kWh)": y_test, "Predicted Energy (kWh)": predictions}
    )
    print(results.head(10).round(3).to_string(index=False))