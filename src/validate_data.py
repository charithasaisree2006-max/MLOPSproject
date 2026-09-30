from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "raw" / "INVideos.csv"


def main():
    print("Starting data validation...")

    if not DATA_FILE.exists():
        print(f"[ERROR] Dataset not found: {DATA_FILE}")
        return

    df = pd.read_csv(DATA_FILE)

    print(f"[INFO] Dataset loaded successfully")
    print(f"[INFO] Rows: {len(df)}")
    print(f"[INFO] Columns: {len(df.columns)}")

    print("\n[INFO] Column names:")
    for column in df.columns:
        print(f" - {column}")

    print("\n[INFO] Missing values:")
    print(df.isnull().sum())

    print("\n[INFO] Duplicate rows:", df.duplicated().sum())

    if len(df) == 0:
        print("[ERROR] Dataset is empty.")
        return

    print("\n[SUCCESS] Data validation completed successfully.")


if __name__ == "__main__":
    main()