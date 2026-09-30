from pathlib import Path
import json
import joblib
import mlflow
import mlflow.sklearn


# Project root
ROOT = Path(__file__).resolve().parent.parent

# Files
SUMMARY_FILE = ROOT / "artifacts" / "best_model_summary.json"
MODEL_FILE = ROOT / "models" / "best_model.pkl"

# MLflow database
MLFLOW_DB = ROOT / "mlflow.db"


def main():

    print("========================================")
    print("MLflow Experiment Tracking")
    print("========================================")

    # ------------------------------------------------
    # 1. Check required files
    # ------------------------------------------------

    if not SUMMARY_FILE.exists():
        print("[ERROR] best_model_summary.json not found.")
        print("Run train.py first.")
        return

    if not MODEL_FILE.exists():
        print("[ERROR] best_model.pkl not found.")
        print("Run train.py first.")
        return

    # ------------------------------------------------
    # 2. Load model and summary
    # ------------------------------------------------

    print("\n[1] Loading trained model...")

    with open(SUMMARY_FILE, "r", encoding="utf-8") as f:
        summary = json.load(f)

    model = joblib.load(MODEL_FILE)

    print("Model:", summary["best_model"])

    # ------------------------------------------------
    # 3. Configure MLflow
    # ------------------------------------------------

    print("\n[2] Configuring MLflow...")

    mlflow.set_tracking_uri(
        f"sqlite:///{MLFLOW_DB}"
    )

    mlflow.set_experiment("YouTube Views MLOps")

    # ------------------------------------------------
    # 4. Start MLflow run
    # ------------------------------------------------

    print("\n[3] Starting MLflow run...")

    with mlflow.start_run(
        run_name=summary["best_model"]
    ) as run:

        # Model parameter
        mlflow.log_param(
            "model",
            summary["best_model"]
        )

        # Model parameters
        mlflow.log_param(
            "n_estimators",
            summary.get("n_estimators", 200)
        )

        mlflow.log_param(
            "random_state",
            summary.get("random_state", 42)
        )

        # Number of features
        mlflow.log_param(
            "n_features",
            len(summary["features"])
        )

        # Target
        mlflow.log_param(
            "target",
            summary["target"]
        )

        # Metrics
        metrics = summary["metrics"]

        mlflow.log_metric(
            "MAE",
            metrics["mae"]
        )

        mlflow.log_metric(
            "MSE",
            metrics["mse"]
        )

        mlflow.log_metric(
            "RMSE",
            metrics["rmse"]
        )

        mlflow.log_metric(
            "R2",
            metrics["r2"]
        )

        # Log model
        mlflow.sklearn.log_model(
            model,
            "model"
        )

        # Save run ID
        run_id_file = ROOT / "artifacts" / "latest_run_id.txt"

        run_id_file.write_text(
            run.info.run_id,
            encoding="utf-8"
        )

        print("\n========================================")
        print("[SUCCESS] MLflow tracking completed")
        print("========================================")

        print("Run ID:", run.info.run_id)


if __name__ == "__main__":
    main()