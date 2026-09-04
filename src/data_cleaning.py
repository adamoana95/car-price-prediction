import re
import pandas as pd

RAW_DATA_PATH = "data/cars.csv"
CLEANED_DATA_PATH = "data/cars_cleaned.csv"

df = pd.read_csv(RAW_DATA_PATH)

def _standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
 
    new_columns = []
 
    for col in df.columns:
        clean_col = col.strip().lower()
 
        clean_col = clean_col.replace("(", "_")
        clean_col = clean_col.replace(")", "")
        clean_col = clean_col.replace("-", "_")
        clean_col = clean_col.replace("/", "_")
 
        clean_col = re.sub(r"\s+", "_", clean_col)
        clean_col = re.sub(r"[^a-z0-9_]", "", clean_col)
        clean_col = re.sub(r"_+", "_", clean_col)
        clean_col = clean_col.strip("_")
 
        new_columns.append(clean_col)
 
    df.columns = new_columns
    
    if "mileage_kilometers" in df.columns:
        df = df.rename(columns={"mileage_kilometers": "mileage_km"})

    return df



def _strip_string_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
 
    text_columns = df.select_dtypes(include=["object", "string"]).columns
 
    for col in text_columns:
        df[col] = df[col].astype("string").str.strip()
 
    return df



MISSING_LIKE_VALUES = {
    "",
    " ",
    "nan",
    "NaN",
    "NAN",
    "null",
    "Null",
    "NULL",
    "none",
    "None",
    "NONE",
}


def _replace_missing_like_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
 
    df = df.replace(list(MISSING_LIKE_VALUES), pd.NA)
 
    return df



def _convert_numeric_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
 
    # Columns that should be stored as float
    float_columns = [
        "priceusd",
        "mileage_km",
    ]

    for col in float_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            ).astype("float64")

    # Columns that should be stored as integers
    integer_columns = [
        "year",
        "volume_cm3",
    ]

    for col in integer_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            ).astype("Int64")

    return df



def _clean_categorical_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
 
    categorical_columns = [
        "make",
        "model",
        "condition",
        "fuel_type",
        "color",
        "transmission",
        "drive_unit",
        "segment"
    ]
    
    for col in categorical_columns:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype("string")
                .str.strip()
                .str.lower()
            )

    return df


def _clean_invalid_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Invalid prices

    if "priceusd" in df.columns:

        df.loc[df["priceusd"] <= 0, "priceusd"] = pd.NA

    # Invalid mileage

    # Values above 500,000 km are considered extreme/anomalous.

    # The row is kept and only the mileage value is set to missing.

    if "mileage_km" in df.columns:

        df.loc[df["mileage_km"] < 0, "mileage_km"] = pd.NA

        df.loc[df["mileage_km"] > 500000, "mileage_km"] = pd.NA

    # Invalid production years

    if "year" in df.columns:

        df.loc[
            ~df["year"].between(1900, 2019),
            "year"
        ] = pd.NA

    # Invalid engine volumes

    if "volume_cm3" in df.columns:

        df.loc[df["volume_cm3"] <= 0, "volume_cm3"] = pd.NA

    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:

    df = _standardize_column_names(df)

    df = _strip_string_values(df)

    df = _replace_missing_like_values(df)

    df = _convert_numeric_columns(df)

    df = _clean_categorical_values(df)

    df = _clean_invalid_values(df)

    return df

# Apply cleaning

df_cleaned = clean_data(df)

# Save cleaned dataset

df_cleaned.to_csv(
    CLEANED_DATA_PATH,
    index=False
)

# Basic information about the cleaning process

print(f"Original rows: {len(df)}")

print(f"Cleaned rows: {len(df_cleaned)}")

print(f"Removed rows: {len(df) - len(df_cleaned)}")

print(f"Cleaned data saved to: {CLEANED_DATA_PATH}")