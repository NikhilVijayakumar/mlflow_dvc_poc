#src.training.train.py

import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import mlflow
import mlflow.sklearn

from src.config.settings import app_config, env_config
from src.utils.io import save_json

def train():
    print("🚀 Starting Stage: Model Training")

    mlflow.set_tracking_uri(env_config.MLFLOW_TRACKING_URI)
    print(f"   - MLflow Tracking URI: {env_config.MLFLOW_TRACKING_URI}")

    experiment_name = app_config.mlflow.experiment_name
    mlflow.set_experiment(experiment_name)
    print(f"   - MLflow Experiment: '{experiment_name}'")

    train_path = app_config.paths.train_data
    model_output_path = app_config.paths.model
    reports_path = app_config.paths.reports
    training_params = app_config.training

    print(f"   - Loading training data from: {train_path}")
    df_train = pd.read_csv(train_path)
    X_train = df_train.drop(columns=["target"])
    y_train = df_train["target"]

    with mlflow.start_run() as run:
        print(f"   - MLflow Run ID: {run.info.run_id}")

        mlflow.log_params(training_params.dict())
        print(f"   - Logging parameters: {training_params.dict()}")

        if training_params.model_type == "LogisticRegression":
            model = LogisticRegression(
                max_iter=training_params.max_iter,
                random_state=training_params.random_state
            )
        else:
            raise ValueError(f"Unsupported model type: {training_params.model_type}")

        print("   - Fitting model...")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_train)
        accuracy = accuracy_score(y_train, y_pred)

        print(f"   - Training Accuracy: {accuracy:.4f}")
        mlflow.log_metric("train_accuracy", accuracy)

        print(f"   - Logging and registering model as '{training_params.registered_model_name}'")
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model", # Subfolder in the MLflow run artifacts
            registered_model_name=training_params.registered_model_name
        )

    print(f"   - Saving model for DVC to: {model_output_path}")
    with open(model_output_path, "wb") as f:
        pickle.dump(model, f)

    metrics = {"train_accuracy": accuracy}
    save_json(metrics, reports_path)

    print("🏁 Finished Stage: Model Training")


if __name__ == "__main__":
    train()

