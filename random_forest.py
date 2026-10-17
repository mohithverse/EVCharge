"""Predict EV charging energy with a random forest."""

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from tree_models_utils import build_preprocessor, load_data, report_results


def main() -> None:
    X_train, X_test, y_train, y_test = load_data()
    model = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=100,
                    max_depth=16,
                    min_samples_leaf=5,
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    report_results("Random Forest Regression", model, X_test, y_test)


if __name__ == "__main__":
    main()