#src.training.train.py

import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    classification_report
)
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.data.pandas_dataset import from_pandas

from src.config.settings import app_config, env_config
from src.utils.io import save_json


class ExperimentRunner:

    def __init__(self):
        print("🚀 Initializing Experiment Runner...")
        self.paths = app_config.paths
        self.params = app_config.training
        self.mlflow_config = app_config.mlflow

        mlflow.set_tracking_uri(env_config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(self.mlflow_config.experiment_name)

        self.model = None
        self.X_train, self.y_train = None, None
        self.X_test, self.y_test = None, None
        self.test_preds = None
        self.df_train = pd.read_csv(self.paths.train_data)
        self.df_test = pd.read_csv(self.paths.test_data)

    def _log_datasets(self):
        print("   - Logging dataset URIs to MLflow...")

        train_dataset = from_pandas(self.df_train, source=self.paths.train_data)
        test_dataset = from_pandas(self.df_test, source=self.paths.test_data)

        # Log the datasets to the current run with a clear context
        mlflow.log_input(train_dataset, context="training_data")
        mlflow.log_input(test_dataset, context="testing_data")

    def _load_data(self):
        print("   - Loading data...")
        self.X_train, self.y_train = self.df_train.drop(columns=["target"]), self.df_train["target"]
        self.X_test, self.y_test = self.df_test.drop(columns=["target"]), self.df_test["target"]

    def _train_model(self):
        print("   - Fitting model...")
        self.model = LogisticRegression(
            max_iter=self.params.max_iter,
            random_state=self.params.random_state
        )
        self.model.fit(self.X_train, self.y_train)

    def _evaluate_and_log_metrics(self):
        print("   - Evaluating model and logging metrics...")
        test_preds = self.model.predict(self.X_test)
        test_probas = self.model.predict_proba(self.X_test)

        test_accuracy = accuracy_score(self.y_test, test_preds)
        test_precision = precision_score(self.y_test, test_preds, average='weighted')
        test_recall = recall_score(self.y_test, test_preds, average='weighted')
        test_f1 = f1_score(self.y_test, test_preds, average='weighted')
        test_log_loss = log_loss(self.y_test, test_probas)
        test_roc_auc = roc_auc_score(self.y_test, test_probas, multi_class='ovr', average='weighted')

        print(f"   - Test Accuracy: {test_accuracy:.4f}")

        mlflow.log_metric("test_accuracy", test_accuracy)
        mlflow.log_metric("test_precision", test_precision)
        mlflow.log_metric("test_recall", test_recall)
        mlflow.log_metric("test_f1_score", test_f1)
        mlflow.log_metric("test_log_loss", test_log_loss)
        mlflow.log_metric("test_roc_auc", test_roc_auc)

        test_report_dict = classification_report(self.y_test, test_preds, output_dict=True)
        final_metrics = {
            "test_accuracy": test_accuracy,
            "test_f1_score": test_f1,
            "test_roc_auc": test_roc_auc,
            "classification_report": test_report_dict
        }
        save_json(final_metrics, self.paths.reports)

        mlflow.log_dict(test_report_dict, "classification_report.json")

    def _log_and_register_model(self,run_id):
        print(f"   - Logging and registering model as '{self.params.registered_model_name}'...")

        # ✅ Create the signature and input example
        signature = infer_signature(self.X_test, self.test_preds)
        input_example = self.X_test.head()

        mlflow.sklearn.log_model(
            sk_model=self.model,
            artifact_path="classifier",
            signature=signature,
            input_example=input_example
        )

        client = mlflow.MlflowClient()
        model_uri = f"runs:/{run_id}/model"

        new_version = client.create_model_version(
            name=self.params.registered_model_name,
            source=model_uri,
            run_id=run_id,
            description=self.mlflow_config.registered_model_description,
            tags=self.mlflow_config.model_version_tags
        )

        print(f"   - ✅ Registered model version {new_version.version}")

        with open(self.paths.model, "wb") as f:
            pickle.dump(self.model, f)

    def run(self):
        self._load_data()

        with mlflow.start_run() as run:
            print(f"   - MLflow Run ID: {run.info.run_id}")
            mlflow.log_params(self.params.dict())
            self._log_datasets()
            self._train_model()
            self._evaluate_and_log_metrics()
            self._log_and_register_model(run.info.run_id)

        print("🏁 Finished Stage: Run Full Experiment")


if __name__ == "__main__":
    runner = ExperimentRunner()
    runner.run()

