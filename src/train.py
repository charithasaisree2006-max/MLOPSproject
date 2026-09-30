from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Project root
ROOT = Path(__file__).resolve().parent.parent

# Files
DATA_FILE = ROOT / "data" / "processed" / "processed_data.csv"
MODEL_DIR = ROOT / "models"
ARTIFACT_DIR = ROOT / "artifacts"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def find_target_column(df):
    """
    Find the YouTube views column automatically.
    """

    possible_targets = [
        "views",
        "view_count",
        "viewcount",
        "views_count",
        "video_views",
        "Views",
        "ViewCount"
    ]

    for column in possible_targets:
        if column in df.columns:
            return column

    return None


def main():

    print("========================================")
    print("YouTube Views Model Training")
    print("========================================")

    # ------------------------------------------------
    # 1. Load processed data
    # ------------------------------------------------

    print("\n[1] Loading processed dataset...")

    df = pd.read_csv(DATA_FILE)

    print("Dataset shape:", df.shape)

    # ------------------------------------------------
    # 2. Find target column
    # ------------------------------------------------

    target_column = find_target_column(df)

    if target_column is None:
        print("\n[ERROR] Views target column was not found.")
        print("\nAvailable columns:")

        for column in df.columns:
            print("-", column)

        print("\nPlease check the target column name.")
        return

    print("\nTarget column:", target_column)

    # ------------------------------------------------
    # 3. Prepare data
    # ------------------------------------------------

    print("\n[2] Preparing features...")

    # Keep only numeric columns
    numeric_df = df.select_dtypes(include=["number"]).copy()

    if target_column not in numeric_df.columns:
        print("[ERROR] Target column is not numeric.")
        return

    # Remove target from features
    X = numeric_df.drop(columns=[target_column])
    y = numeric_df[target_column]

    # Remove rows with missing values
    valid_rows = X.notna().all(axis=1) & y.notna()

    X = X.loc[valid_rows]
    y = y.loc[valid_rows]

    print("Features:", list(X.columns))
    print("Number of features:", X.shape[1])
    print("Number of samples:", len(X))

    if X.shape[1] == 0:
        print("[ERROR] No numeric features available.")
        return

    # ------------------------------------------------
    # 4. Split dataset
    # ------------------------------------------------

    print("\n[3] Splitting dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # ------------------------------------------------
    # 5. Train Random Forest
    # ------------------------------------------------

    print("\n[4] Training Random Forest...")

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    print("[SUCCESS] Model training completed.")

    # ------------------------------------------------
    # 6. Prediction
    # ------------------------------------------------

    print("\n[5] Evaluating model...")

    y_pred = model.predict(X_test)

    # ------------------------------------------------
    # 7. Metrics
    # ------------------------------------------------

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    print("\n========================================")
    print("MODEL PERFORMANCE")
    print("========================================")

    print("MAE  :", mae)
    print("MSE  :", mse)
    print("RMSE :", rmse)
    print("R2   :", r2)

    # ------------------------------------------------
    # 8. Save model
    # ------------------------------------------------

    model_file = MODEL_DIR / "best_model.pkl"

    joblib.dump(model, model_file)

    print("\n[6] Model saved:")
    print(model_file)

    # ------------------------------------------------
    # 9. Save model summary
    # ------------------------------------------------

    summary = {
        "best_model": "RandomForestRegressor",
        "target": target_column,
        "features": list(X.columns),
        "metrics": {
            "mae": float(mae),
            "mse": float(mse),
            "rmse": float(rmse),
            "r2": float(r2)
        },
        "random_state": 42,
        "n_estimators": 200
    }

    summary_file = ARTIFACT_DIR / "best_model_summary.json"

    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print("\n[7] Model summary saved:")
    print(summary_file)

    # ------------------------------------------------
    # 10. Save test results
    # ------------------------------------------------

    results = pd.DataFrame({
        "actual": y_test.values,
        "predicted": y_pred
    })

    results_file = ARTIFACT_DIR / "test_predictions.csv"

    results.to_csv(results_file, index=False)

    print("\n[8] Test predictions saved:")
    print(results_file)

    print("\n========================================")
    print("[SUCCESS] MODEL TRAINING COMPLETED")
    print("========================================")


if __name__ == "__main__":
    main()