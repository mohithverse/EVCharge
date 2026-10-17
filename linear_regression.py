"""Train energy-regression and station-busy classification models."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDRegressor
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_FILE = "EVChargingStationUsage_cleaned.csv"
TARGET = "Energy (kWh)"
NUMERIC_FEATURES = ["Charging Time Minutes", "Hour"]
CATEGORICAL_FEATURES = ["Port Type", "Plug Type"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# A station-hour is considered busy when two or more sessions begin at the same
# station during that clock hour. This is an observable demand signal in the
# session dataset (rather than a real-time occupancy feed).
BUSY_NUMERIC_FEATURES = ["Hour", "Month", "Port Number"]
BUSY_CATEGORICAL_FEATURES = [
    "Station Name",
    "Day of Week",
    "Port Type",
    "Plug Type",
]
BUSY_FEATURES = BUSY_NUMERIC_FEATURES + BUSY_CATEGORICAL_FEATURES
BUSY_THRESHOLD = 2


def build_preprocessor(
    numeric_features: list[str], categorical_features: list[str], scale_numeric: bool
) -> ColumnTransformer:
    """Create the transformations required before fitting a model."""
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("numeric", Pipeline(numeric_steps), numeric_features),
            (
                "categorical",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("one_hot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )


def metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    """Return the regression metrics in kWh where applicable."""
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": mean_squared_error(actual, predicted) ** 0.5,
        "R2": r2_score(actual, predicted),
    }


def classification_metrics(
    actual: pd.Series, predicted: np.ndarray, probabilities: np.ndarray
) -> dict[str, float]:
    """Return evaluation metrics for the busy/not-busy classifier."""
    return {
        "Accuracy": accuracy_score(actual, predicted),
        "Precision": precision_score(actual, predicted, zero_division=0),
        "Recall": recall_score(actual, predicted, zero_division=0),
        "F1": f1_score(actual, predicted, zero_division=0),
        "ROC-AUC": roc_auc_score(actual, probabilities),
    }


def prepare_busy_station_data(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    busy_data = data[BUSY_FEATURES + ["Start Date"]].copy()
    busy_data["Start Date"] = pd.to_datetime(busy_data["Start Date"], errors="coerce")
    busy_data = busy_data.dropna(subset=["Start Date", "Station Name"])
    busy_data["station_hour"] = busy_data["Start Date"].dt.floor("h")
    sessions_in_station_hour = busy_data.groupby(
        ["Station Name", "station_hour"]
    )["Station Name"].transform("size")
    busy_target = (sessions_in_station_hour >= BUSY_THRESHOLD).astype(int)
    station_hour_groups = (
        busy_data["Station Name"].astype(str)
        + " | "
        + busy_data["station_hour"].astype(str)
    )
    return busy_data[BUSY_FEATURES], busy_target, station_hour_groups


def main() -> None:
    source_data = pd.read_csv(DATA_FILE, low_memory=False)
    data = source_data[FEATURES + [TARGET]].dropna(subset=[TARGET])

    X = data[FEATURES]
    y = data[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # Closed-form linear regression: the primary model used for predictions.
    linear_model = Pipeline(
        [
            (
                "preprocessor",
                build_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES, scale_numeric=False),
            ),
            ("model", LinearRegression()),
        ]
    )
    linear_model.fit(X_train, y_train)
    linear_predictions = np.maximum(linear_model.predict(X_test), 0)

    # SGDRegressor learns a linear model through gradient descent.
    gradient_descent_model = Pipeline(
        [
            (
                "preprocessor",
                build_preprocessor(NUMERIC_FEATURES, CATEGORICAL_FEATURES, scale_numeric=True),
            ),
            (
                "model",
                SGDRegressor(
                    loss="squared_error",
                    max_iter=2_000,
                    tol=1e-4,
                    learning_rate="adaptive",
                    eta0=0.01,
                    random_state=42,
                ),
            ),
        ]
    )
    gradient_descent_model.fit(X_train, y_train)
    gradient_predictions = np.maximum(gradient_descent_model.predict(X_test), 0)

    linear_scores = metrics(y_test, linear_predictions)
    gradient_scores = metrics(y_test, gradient_predictions)
    model_name, predictions, _ = min(
        [
            ("Linear Regression", linear_predictions, linear_scores),
            ("Gradient Descent", gradient_predictions, gradient_scores),
        ],
        key=lambda result: result[2]["RMSE"],
    )

    print(f"Rows used: {len(data):,}")
    print(f"Training rows: {len(X_train):,} | Test rows: {len(X_test):,}")
    print("\nModel evaluation (lower MAE/RMSE is better):")
    for name, model_scores in (
        ("Linear Regression", linear_scores),
        ("Gradient Descent", gradient_scores),
    ):
        print(
            f"{name}: MAE={model_scores['MAE']:.3f} kWh, "
            f"RMSE={model_scores['RMSE']:.3f} kWh, R²={model_scores['R2']:.3f}"
        )

    print(f"\nBest prediction model: {model_name}")
    comparison = pd.DataFrame(
        {"Actual Energy (kWh)": y_test.to_numpy(), "Predicted Energy (kWh)": predictions}
    )
    print("\nSample predictions:")
    print(comparison.head(10).round(3).to_string(index=False))

    # Logistic regression: classify whether a station-hour is busy or not.
    busy_X, busy_y, station_hour_groups = prepare_busy_station_data(source_data)
    busy_train_index, busy_test_index = next(
        GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42).split(
            busy_X, busy_y, groups=station_hour_groups
        )
    )
    busy_X_train, busy_X_test = (
        busy_X.iloc[busy_train_index],
        busy_X.iloc[busy_test_index],
    )
    busy_y_train, busy_y_test = (
        busy_y.iloc[busy_train_index],
        busy_y.iloc[busy_test_index],
    )
    busy_model = Pipeline(
        [
            (
                "preprocessor",
                build_preprocessor(
                    BUSY_NUMERIC_FEATURES,
                    BUSY_CATEGORICAL_FEATURES,
                    scale_numeric=True,
                ),
            ),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced", max_iter=500, random_state=42
                ),
            ),
        ]
    )
    busy_model.fit(busy_X_train, busy_y_train)
    busy_predictions = busy_model.predict(busy_X_test)
    busy_probabilities = busy_model.predict_proba(busy_X_test)[:, 1]
    busy_scores = classification_metrics(
        busy_y_test, busy_predictions, busy_probabilities
    )

    print("\nBusy-station classification (Logistic Regression):")
    print(
        "Busy definition: at least "
        f"{BUSY_THRESHOLD} recorded sessions start at the same station in one clock hour."
    )
    print(
        f"Training rows: {len(busy_X_train):,} | Test rows: {len(busy_X_test):,} | "
        f"Busy rate: {busy_y.mean():.1%}"
    )
    print(
        "Accuracy={Accuracy:.1%}, Precision={Precision:.1%}, Recall={Recall:.1%}, "
        "F1={F1:.3f}, ROC-AUC={ROC-AUC:.3f}".format(**busy_scores)
    )


if __name__ == "__main__":
    main()
