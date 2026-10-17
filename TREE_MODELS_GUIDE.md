# Tree-Based Energy Models

All four scripts predict `Energy (kWh)` from `Charging Time Minutes`, `Hour`,
`Port Type`, and `Plug Type` in `EVChargingStationUsage_cleaned.csv`.

## Concepts

- **Decision Tree**: learns a sequence of if/else splits. It is easy to
  interpret, but a single deep tree can overfit.
- **Random Forest**: trains many decision trees on random samples and averages
  their predictions. This usually reduces variance and overfitting.
- **AdaBoost**: trains small trees sequentially, giving more attention to rows
  that earlier trees predicted poorly.
- **XGBoost**: builds regularized gradient-boosted trees efficiently, where each
  new tree corrects errors from the previous trees.

## Run

```powershell
python decision_tree.py
python random_forest.py
python adaboost.py
python -m pip install xgboost
python xgboost_model.py
```

Each script prints MAE, RMSE, R², and sample predictions. Lower MAE/RMSE and
higher R² indicate better regression performance on the held-out test set.