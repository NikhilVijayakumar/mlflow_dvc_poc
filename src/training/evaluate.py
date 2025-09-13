#src.training.evaluate.py

import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn

from src.config.settings import app_config, env_config
from src.utils.io import save_json

def evaluate():
    print("🚀 Starting Stage: Run Full Experiment (Train & Evaluate)")

    mlflow.set_tracking_uri(env_config.MLFLOW_TRACKING_URI)
    experiment_name = app_config.mlflow.experiment_name
    mlflow.set_experiment(experiment_name)
    print(f"   - MLflow Experiment: '{experiment_name}'")

    paths = app_config.paths
    params = app_config.training

    print(f"   - Loading data from: {paths.train_data} and {paths.test_data}")
    df_train = pd.read_csv(paths.train_data)
    df_test = pd.read_csv(paths.test_data)

    X_train, y_train = df_train.drop(columns=["target"]), df_train["target"]
    X_test, y_test = df_test.drop(columns=["target"]), df_test["target"]

    with mlflow.start_run() as run:
        print(f"   - MLflow Run ID: {run.info.run_id}")

        mlflow.log_params(params.dict())
        print(f"   - Logging parameters: {params.dict()}")

        model = LogisticRegression(
            max_iter=params.max_iter,
            random_state=params.random_state
        )
        print("   - Fitting model...")
        model.fit(X_train, y_train)

        print("   - Evaluating model on test set...")
        y_pred = model.predict(X_test)

        test_accuracy = accuracy_score(y_test, y_pred)
        report_dict = classification_report(y_test, y_pred, output_dict=True)

        print(f"   - Test Accuracy: {test_accuracy:.4f}")
        mlflow.log_metric("test_accuracy", test_accuracy)

        mlflow.log_metric("precision_class_1", report_dict["1"]["precision"])
        mlflow.log_metric("recall_class_1", report_dict["1"]["recall"])

        print(f"   - Logging and registering model as '{params.registered_model_name}'")
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=params.registered_model_name
        )

    print(f"   - Saving final model for DVC to: {paths.model}")
    with open(paths.model, "wb") as f:
        pickle.dump(model, f)

    final_metrics = {
        "test_accuracy": test_accuracy,
        "classification_report": report_dict
    }
    save_json(final_metrics, paths.reports)

    print("🏁 Finished Stage: Run Full Experiment")

if __name__ == "__main__":
    evaluate()

