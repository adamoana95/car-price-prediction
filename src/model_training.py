import joblib
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from data_preprocessing import (
    split_features_and_target,
    build_preprocessor,
)


DATA_PATH = "data/cars_features.csv"
MODEL_PATH = "models/linear_regression_model.joblib"


print("Loading dataset...")

df = pd.read_csv(DATA_PATH)


print("Splitting features and target...")

X, y = split_features_and_target(df)

print(X.shape)
print(y.shape)


print("Splitting data into training and test sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)


print("Creating model pipeline...")

model = Pipeline(
    steps=[
        ("preprocessor", build_preprocessor()),
        ("regressor", LinearRegression()),
    ]
)


print("Training model...")

model.fit(X_train, y_train)


print("Saving model...")

joblib.dump(model, MODEL_PATH)

print(f"Model saved to: {MODEL_PATH}")


print("Loading saved model...")

loaded_model = joblib.load(MODEL_PATH)


print("Making sample predictions...")

sample_X = X_test.sample(10, random_state=42)
sample_y = y_test.loc[sample_X.index]

sample_predictions = loaded_model.predict(sample_X)


prediction_preview = pd.DataFrame({
    "actual_price_usd": sample_y.values,
    "predicted_price_usd": sample_predictions,
})


print(prediction_preview)