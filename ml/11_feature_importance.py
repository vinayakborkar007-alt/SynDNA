import pandas as pd
import joblib

X_train = pd.read_csv("dataset/X_train_reduced.csv")

model = joblib.load("ml/xgboost_reduced_model.pkl")

importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "feature": X_train.columns,
    "importance": importance
})

feature_importance = feature_importance.sort_values(
    by="importance",
    ascending=False
)

print("SynDNA - Feature Importance")
print("=" * 40)

print("\nTop 20 features:")
print(
    feature_importance.head(20).to_string(index=False)
)

feature_importance.to_csv(
    "ml/feature_importance.csv",
    index=False
)

print("\nSaved:")
print("ml/feature_importance.csv")

print("\nFeature importance analysis completed!")