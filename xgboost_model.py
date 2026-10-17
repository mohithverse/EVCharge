"""Predict EV charging energy with XGBoost gradient-boosted trees."""

from sklearn.pipeline import Pipeline

from tree_models_utils import build_preprocessor, load_data, report_results

try:
    from xgboost import XGBRegressor
except ImportError as error:
    raise SystemExit(
        "XGBoost is not installed. Run: python -m pip install xgboost"
    ) from error


def main() -> None:
    X_train, X_test, y_train, y_test = load_data()
    model = Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            (
                "model",
                XGBRegressor(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="reg:squarederror",
                    eval_metric="rmse",
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)
    report_results("XGBoost Regression", model, X_test, y_test)


if __name__ == "__main__":
    main()