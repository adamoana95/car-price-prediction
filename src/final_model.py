import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from data_preprocessing import (
    split_features_and_target,
    build_preprocessor,
)

DATA_PATH = "data/cars_features.csv"
MODEL_PATH = "models/random_forest_model.joblib"

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

print("Splitting features and target...")
X, y = split_features_and_target(df)

print("Splitting data into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

print("Creating final Random Forest model...")

model = Pipeline(
    steps=[
        ("preprocessor", build_preprocessor()),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=50,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)

print("Training final model...")
model.fit(X_train, y_train)

print("Making predictions...")
y_pred = model.predict(X_test)

print("Calculating final metrics...")

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nFinal model: Random Forest")
print(f"MAE:  {mae:.2f}")
print(f"MSE:  {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2:   {r2:.4f}")

print("\nSaving final model...")
joblib.dump(model, MODEL_PATH)

print(f"Final model saved to: {MODEL_PATH}")


# Pe baza rezultatelor obținute în etapa de comparare a algoritmilor, modelul Random Forest Regressor a fost ales ca model final. Acesta a obținut cele mai bune rezultate dintre modelele testate, având cel mai mic MAE (1058.74 USD) și cel mai mic RMSE (2631.96 USD), precum și cel mai mare coeficient R² (0.8943). Prin urmare, Random Forest oferă cele mai precise predicții pentru prețul mașinilor second-hand dintre modelele analizate.