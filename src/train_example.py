import pickle
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import dvc.api

# ---------------------------
# Load dataset
# ---------------------------
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)

# ---------------------------
# Train model
# ---------------------------
params = {"C": 1.0, "max_iter": 200, "solver": "lbfgs", "multi_class": "auto"}
model = LogisticRegression(**params)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# ---------------------------
# MLflow logging
# ---------------------------
with mlflow.start_run() as run:
    # log params
    mlflow.log_params(params)

    # log metrics
    mlflow.log_metric("accuracy", accuracy)

    # log dataset info from DVC (if tracked)
    try:
        dataset_path = dvc.api.get_url("data/iris.csv")
        dataset_rev = dvc.api.get_rev("data/iris.csv")
        mlflow.log_param("dataset_path", dataset_path)
        mlflow.log_param("dataset_rev", dataset_rev)
    except Exception as e:
        mlflow.log_param("dataset", "builtin_sklearn_iris")
        print("⚠️ Dataset not tracked in DVC, falling back to sklearn iris dataset")

    # log + register model in MLflow
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="LogisticRegressionModel"
    )

    print(f"✅ Run {run.info.run_id} logged to MLflow")
    print(f"Accuracy: {accuracy:.4f}")

# ---------------------------
# Save model for DVC versioning
# ---------------------------
with open("models/log_reg_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model saved to models/log_reg_model.pkl for DVC tracking")
