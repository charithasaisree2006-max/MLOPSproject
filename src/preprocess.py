from pathlib import Path
import pandas as pd
import numpy as np
import joblib

ROOT = Path(__file__).resolve().parent.parent

RAW_FILE = ROOT / "data" / "raw" / "INVideos.csv"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def preprocess():
    print("\n========================================")
    print("YouTube Views Preprocessing")
    print("========================================")

    # Load dataset
    print("\n[1] Loading dataset...")
    df = pd.read_csv(RAW_FILE)

    print("Dataset shape:", df.shape)

    # Remove duplicate rows
    print("\n[2] Removing duplicate rows...")
    before = len(df)
    df = df.drop_duplicates()
    after = len(df)

    print("Duplicates removed:", before - after)

    # Remove rows with missing values
    print("\n[3] Handling missing values...")
    df = df.dropna()

    print("Shape after removing missing values:", df.shape)

    # Convert object columns to numeric where possible
    print("\n[4] Converting data types...")

    for column in df.columns:
        if df[column].dtype == "object":
            try:
                df[column] = pd.to_numeric(df[column])
            except:
                pass

    # Save processed dataset
    processed_file = PROCESSED_DIR / "processed_data.csv"

    df.to_csv(processed_file, index=False)

    print("\n[5] Processed dataset saved:")
    print(processed_file)

    # Save preprocessing information
    info = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns)
    }

    joblib.dump(info, MODEL_DIR / "preprocessing_info.pkl")

    print("\n[6] Preprocessing information saved.")

    print("\n========================================")
    print("[SUCCESS] Preprocessing completed.")
    print("========================================")


if __name__ == "__main__":
    preprocess()