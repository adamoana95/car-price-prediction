import pandas as pd


CLEANED_DATA_PATH = "data/cars_cleaned.csv"
FEATURES_DATA_PATH = "data/cars_features.csv"

REFERENCE_YEAR = 2026


def _create_car_age(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["car_age"] = REFERENCE_YEAR - df["year"]

    return df


def _create_mileage_per_year(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["mileage_per_year"] = (
        df["mileage_km"]
        / df["car_age"].replace(0, pd.NA)
    )

    return df


def _create_engine_volume_liters(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["engine_volume_liters"] = df["volume_cm3"] / 1000

    return df


def _create_is_newer_car(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["is_newer_car"] = (
        df["year"] >= 2010
    ).map({True: "yes", False: "no"})

    return df


def _create_is_high_mileage(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["is_high_mileage"] = (
        df["mileage_km"] >= 300000
    ).map({True: "yes", False: "no"})

    return df


def _create_brand_model(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["brand_model"] = (
        df["make"].fillna("unknown")
        + "_"
        + df["model"].fillna("unknown")
    )

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df_features = (
        df
        .pipe(_create_car_age)
        .pipe(_create_mileage_per_year)
        .pipe(_create_engine_volume_liters)
        .pipe(_create_is_newer_car)
        .pipe(_create_is_high_mileage)
        .pipe(_create_brand_model)
        .reset_index(drop=True)
    )

    return df_features


def main() -> None:
    """Load cleaned data, build features, and save the feature-engineered dataset."""

    print("Loading cleaned dataset...")

    df_cleaned = pd.read_csv(CLEANED_DATA_PATH)

    print("Building features...")

    df_features = build_features(df_cleaned)

    print("Saving feature-engineered dataset...")

    df_features.to_csv(FEATURES_DATA_PATH, index=False)

    print(f"Feature-engineered dataset saved to: {FEATURES_DATA_PATH}")


if __name__ == "__main__":
    main()