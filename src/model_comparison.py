import pandas as pd

from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

from data_preprocessing import (
    split_features_and_target,
    build_preprocessor,
)


DATA_PATH = "data/cars_features.csv"


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


models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(
        random_state=42
    ),
    "Random Forest": RandomForestRegressor(
        n_estimators=50,
        random_state=42,
        n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        random_state=42
    ),
}


results = []


for model_name, regressor in models.items():

    print(f"\nTraining {model_name}...")

    model = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("regressor", regressor),
        ]
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)

    mse = mean_squared_error(y_test, y_pred)

    rmse = mse ** 0.5

    r2 = r2_score(y_test, y_pred)

    results.append({
        "model": model_name,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
    })


results_df = pd.DataFrame(results)


results_df = results_df.sort_values(
    by="mae",
    ascending=True,
)


print("\nModel comparison:")

print(results_df)


# În urma comparării celor patru algoritmi de regresie, modelul Random Forest a obținut cele mai bune rezultate. Acesta a avut cel mai mic MAE, de aproximativ 1.059 USD, cel mai mic RMSE, de aproximativ 2.632 USD, și cel mai mare coeficient R², de aproximativ 0.894. Prin urmare, Random Forest a fost ales ca model cu cea mai bună performanță pentru predicția prețului mașinilor second-hand.