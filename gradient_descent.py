"""Predict EV charging energy with batch gradient descent implemented from scratch."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


DATA_FILE = "EVChargingStationUsage_cleaned.csv"
TARGET = "Energy (kWh)"
FEATURES = ["Charging Time Minutes", "Hour"]
LEARNING_RATE = 0.05
ITERATIONS = 2_000


def standardize(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Scale features using only training-data statistics."""
    mean = train.mean(axis=0)
    std = train.std(axis=0)
    std[std == 0] = 1
    return (train - mean) / std, (test - mean) / std


def fit_gradient_descent(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit linear-regression weights using batch gradient descent."""
    weights = np.zeros(X.shape[1])
    bias = 0.0
    row_count = len(X)

    for _ in range(ITERATIONS):
        errors = X @ weights + bias - y
        weights -= LEARNING_RATE * (X.T @ errors) / row_count
        bias -= LEARNING_RATE * errors.mean()

    return weights, bias


def main() -> None:
    data = pd.read_csv(DATA_FILE, usecols=FEATURES + [TARGET], low_memory=False)
    data = data.dropna()

    X_train, X_test, y_train, y_test = train_test_split(
        data[FEATURES].to_numpy(dtype=float),
        data[TARGET].to_numpy(dtype=float),
        test_size=0.20,
        random_state=42,
    )
    X_train, X_test = standardize(X_train, X_test)
    weights, bias = fit_gradient_descent(X_train, y_train)

    predictions = np.maximum(X_test @ weights + bias, 0)
    print(f"Rows used: {len(data):,}")
    print("\nGradient Descent Energy Prediction Results")
    print(f"MAE:  {mean_absolute_error(y_test, predictions):.3f} kWh")
    print(f"RMSE: {mean_squared_error(y_test, predictions) ** 0.5:.3f} kWh")
    print(f"R²:   {r2_score(y_test, predictions):.3f}")
    print("\nLearned coefficients:")
    for feature, weight in zip(FEATURES, weights):
        print(f"{feature}: {weight:.3f}")
    print(f"Intercept: {bias:.3f}")

    results = pd.DataFrame(
        {"Actual Energy (kWh)": y_test, "Predicted Energy (kWh)": predictions}
    )
    print("\nSample predictions:")
    print(results.head(10).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
