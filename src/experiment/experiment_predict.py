# src.experiment.experiment_predict.py
import getpass

import mlflow
import pandas as pd
from pathlib import Path

from src.config.settings import app_config, env_config


class ExperimentPredict:

    def __init__(self):
        print("🚀 Initializing Prediction Runner...")
        self.paths = app_config.paths
        self.pred_config = app_config.prediction
        self.mlflow_config = app_config.mlflow

        mlflow.set_tracking_uri(env_config.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(self.mlflow_config.prediction_experiment_name)

    def _load_model_from_registry(self):
        model_uri = f"models:.{self.pred_config.model_name}.{self.pred_config.model_stage}"
        print(f"   - Loading model from URI: {model_uri}")
        try:
            return mlflow.pyfunc.load_model(model_uri)
        except Exception as e:
            print(f"❌ Failed to load model. Have you registered a model version first?")
            print(f"   Error: {e}")
            exit(1)

    def run(self):
        model = self._load_model_from_registry()

        print(f"   - Loading data for prediction from: {self.paths.test_data}")
        df_to_predict = pd.read_csv(self.paths.test_data).drop(columns=["target"], errors='ignore')

        with mlflow.start_run() as run:
            print(f"   - Logging prediction results to MLflow Run ID: {run.info.run_id}")
            mlflow.log_param("source_model_uri",
                             f"models:.{self.pred_config.model_name}.{self.pred_config.model_stage}")

            print("   - Running predictions...")
            predictions = model.predict(df_to_predict)
            df_to_predict["prediction"] = predictions

            output_path = Path(self.pred_config.output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df_to_predict.to_csv(output_path, index=False)

            print(f"   - ✅ Predictions saved to: {output_path}")
            mlflow.set_tag("user", getpass.getuser())
            mlflow.log_artifact(self.pred_config.output_path)
            mlflow.log_metric("num_predictions", len(df_to_predict))
        print("🏁 Finished Prediction Run")


if __name__ == "__main__":
    runner = ExperimentPredict()
    runner.run()