import joblib
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split

from data_preprocessing import split_features_and_target


DATA_PATH = "data/cars_features.csv"
MODEL_PATH = "models/linear_regression_model.joblib"


print("Loading dataset...")

df = pd.read_csv(DATA_PATH)


print("Splitting features and target...")

X, y = split_features_and_target(df)


print("Creating the same train/test split...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)


print("Loading trained model...")

model = joblib.load(MODEL_PATH)


print("Making predictions...")

y_pred = model.predict(X_test)

print(y_pred[:10])


print("Calculating regression metrics...")

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = mse ** 0.5

r2 = r2_score(y_test, y_pred)


metrics = pd.DataFrame({
    "metric": ["MAE", "MSE", "RMSE", "R2"],
    "value": [mae, mse, rmse, r2],
})


print("\nRegression metrics:")

print(metrics)


print("\nCreating prediction analysis table...")

prediction_analysis = pd.DataFrame({
    "actual_price_usd": y_test.values,
    "predicted_price_usd": y_pred,
})


prediction_analysis["error_usd"] = (
    prediction_analysis["actual_price_usd"]
    - prediction_analysis["predicted_price_usd"]
)


prediction_analysis["absolute_error_usd"] = (
    prediction_analysis["error_usd"].abs()
)


print("\nPrediction examples...")

print(
    prediction_analysis
    .sample(10, random_state=42)
)


print("\nLargest prediction errors...")

print(
    prediction_analysis
    .sort_values("absolute_error_usd", ascending=False)
    .head(10)
)



# Regresia liniară a obținut un MAE de aproximativ 1,989 USD, ceea ce înseamnă că, în medie, predicțiile diferă de valorile reale cu aproximativ 1,989 USD. RMSE-ul de aproximativ 4,193 USD este mai mare decât MAE din cauza unor erori foarte mari, în special în cazul automobilelor cu prețuri ridicate. Coeficientul R² este 0.732, ceea ce indică faptul că modelul explică aproximativ 73.2% din variația prețurilor din setul de test. Rezultatele arată că regresia liniară reprezintă un punct de plecare rezonabil, dar există spațiu pentru îmbunătățirea performanței, în special pentru automobilele foarte scumpe.

