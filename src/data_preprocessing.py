import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


FEATURES_DATA_PATH = "data/cars_features.csv"


TARGET_COLUMN = "priceusd"


NUMERIC_FEATURES = [
    "year",
    "mileage_km",
    "volume_cm3",
    "car_age",
    "mileage_per_year",
    "engine_volume_liters",
]


CATEGORICAL_FEATURES = [
    "make",
    "model",
    "condition",
    "fuel_type",
    "color",
    "transmission",
    "drive_unit",
    "segment",
    "is_newer_car",
    "is_high_mileage",
    "brand_model",
]


def get_all_feature_columns() -> list[str]:
    return NUMERIC_FEATURES + CATEGORICAL_FEATURES


def split_features_and_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:

    X = df[get_all_feature_columns()].copy()
    y = df[TARGET_COLUMN].copy()

    return X, y


def _build_numeric_transformer() -> Pipeline:

    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    missing_values=pd.NA,
                    strategy="median",
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )

    return numeric_transformer


def _build_categorical_transformer() -> Pipeline:

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    missing_values=pd.NA,
                    strategy="most_frequent",
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return categorical_transformer


def build_preprocessor() -> ColumnTransformer:

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                _build_numeric_transformer(),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                _build_categorical_transformer(),
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    return preprocessor


def main() -> None:

    print("Loading feature-engineered dataset...")

    df = pd.read_csv(FEATURES_DATA_PATH)

    print("Splitting features and target...")

    X, y = split_features_and_target(df)

    print("Building preprocessing pipeline...")

    preprocessor = build_preprocessor()

    print("Features shape:", X.shape)
    print("Target shape:", y.shape)

    print("Preprocessing completed.")


if __name__ == "__main__":
    main()