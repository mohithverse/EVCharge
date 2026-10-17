"""Predict EV charging energy with AdaBoost regression."""

from sklearn.ensemble import AdaBoostRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from tree_models_utils import build_preprocessor, load_data, report_results


def main() -> None:
    X_train, X_test, y_train, y_test = load_data()
    model = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            (
                "model",
                AdaBoostRegressor(
                    estimator=DecisionTreeRegressor(max_depth=6, min_samples_leaf=10),
                    n_estimators=100,
                    learning_rate=0.05,
                    loss="square",
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    report_results("AdaBoost Regression", model, X_test, y_test)


if __name__ == "__main__":
    main()