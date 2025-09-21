# 🚀 MLOps: ML Pipeline

This repository demonstrates a robust, production-style MLOps workflow for a machine learning project. It is architected around a **single source of truth** for configuration, ensuring reproducibility and maintainability.

The project trains a **Logistic Regression** model on the **Iris dataset** and integrates:

* **MLflow**: For experiment tracking, metric comparison, and model registration.
* **DVC**: For data versioning and creating reproducible ML pipelines.
* **MinIO**: As a self-hosted, S3-compatible backend for DVC artifact storage.
* **Pydantic**: For type-safe, centralized configuration management.

---

## ✨ Core Architecture & Workflow

This project is built on modern MLOps principles, where tools have clear responsibilities and are driven by a central configuration.

* **The 'Single Source of Truth' (`config/config.yaml`)**: This file is the brain of the project. It defines all file paths, hyperparameters, and experiment settings.
    * Our **Python scripts** load this file using a type-safe **Pydantic** settings loader (`src/config/settings.py`).
    * Our **DVC pipeline** (`dvc.yaml`) imports variables directly from this file, making the entire pipeline dynamically configurable.

* **DVC (Data Version Control)**: Manages the pipeline graph and versions data/model artifacts. When you run `dvc repro`, DVC executes the stages defined in `dvc.yaml` in the correct order, using the parameters from `config.yaml`.

* **MLflow**: Acts as the experiment lab book. Our combined training and evaluation stage starts a single MLflow run to log all parameters and metrics, ensuring that each experiment is an **atomic, comparable unit**.

* **MinIO**: Provides a robust and private object storage for all DVC-tracked artifacts (datasets, models, and metrics files).



The workflow is as follows:
1.  The setup script initializes the environment and creates necessary directories.
2.  `dvc repro` executes the pipeline, using `config.yaml` as the master configuration.
3.  The `train_and_evaluate` stage starts a single MLflow run, logging all hyperparameters and both train/test metrics.
4.  DVC versions the final model (`.pkl`) and metrics report (`.json`).
5.  `dvc push` uploads the versioned artifacts to the MinIO server.

---



```

mlflow\_dvc\_poc/
│
├── config/
│   ├── config.yaml          \# 🔥 SINGLE SOURCE OF TRUTH for all configurations
│   └── .env.example         \# Example environment variables
│
├── data/                    \# Raw and processed datasets (tracked by DVC)
├── models/                  \# Trained model artifacts (tracked by DVC)
├── reports/                 \# Final evaluation reports (tracked by DVC)
│
├── src/                     \# All Python source code
│   ├── config/
│   │   ├── settings.py       \# Pydantic settings loader for type-safe config
│   ├── data/                \# Data ingestion and preprocessing scripts
│   │   ├── get_data.py
│   │   ├── preprocess.py
│   ├── experiment/              \# Experiment script 
│   │   ├── experiment_train.py
│   ├── scripts/             \# Helper scripts for setup
│   │   └── setup_dvc_remote.py
│   ├──utils/               \# I/O utility functions
│   │  ├── io.py
│
├── dvc.yaml                 \# DVC pipeline definition (driven by config.yaml)
├── requirements.txt         \# Python dependencies
├── docker-compose.yml       \# Docker service definition for MinIO
└── README.md

````

---

## ⚙️ Setup and Installation

### Prerequisites
* Python 3.12+
* Git
* Docker and Docker Compose

### 1. Clone & Install
```bash
git clone [https://github.com/NikhilVijayakumar/mlflow_dvc_poc.git](https://github.com/NikhilVijayakumar/mlflow_dvc_poc.git)
cd mlflow_dvc_poc
pip install -r requirements.txt
````

### 2\. Configure Secrets

Create a `.env` file for your secrets by copying the example.

```bash
cp config/.env.example config/.env
```

Update `config/.env` with your desired MinIO credentials.

### 3\. Start Backend Services (MinIO & MLflow)

In one terminal, start the MinIO server using Docker Compose:

```bash
docker-compose up -d
```

  * **MinIO Console:** http://localhost:9001

In a second terminal, start the MLflow tracking server:

```bash
mlflow ui
```

  * **MLflow UI:** http://127.0.0.1:5000

### 4\. Initialize the Project Environment

Run the holistic setup script. This command is idempotent and safe to run multiple times.

```bash
python src/scripts/setup_dvc_remote.py
```

This script will:

  * Create all necessary directories (`data/raw`, `models`, etc.).
  * Initialize DVC if needed.
  * Create the MinIO bucket.
  * Configure DVC to use MinIO as remote storage.

-----

## 📊 Running the Pipeline

The DVC pipeline consists of three stages, all dynamically configured by `config/config.yaml`:

1.  `get_data`: Ingests the raw Iris dataset.
2.  `preprocess`: Splits the raw data into training and test sets.
3.  `experiment_runner`: A single, atomic stage that runs a full experiment: trains the model, evaluates it on the test set, and logs everything to a **single MLflow run**.

### Execute the Full Pipeline

To run all stages, use `dvc repro`. DVC will automatically re-run only the stages affected by your changes.

```bash
dvc repro
```

*Tip: To force a re-run of the training stage, simply change a hyperparameter in `config/config.yaml` and run `dvc repro` again.*

### Persist Artifacts to Remote Storage

Push the DVC-tracked artifacts (data, models, metrics) to MinIO.

```bash
dvc push
```

-----

## 📈 Experiment Tracking and Results

### 1\. MLflow UI (http://127.0.0.1:5000)

Navigate to the MLflow UI to compare your experiments. Click on any run to see a complete overview. Because we combined our stages, you will see that **both training and test metrics** (e.g., `test_accuracy`) and all hyperparameters are logged to the **same run**, providing a complete, atomic picture of the experiment.

### 2\. DVC-Tracked Files

The final, version-controlled artifacts are available locally:

  * **Model**: `models/log_reg_model.pkl`
  * **Metrics Report**: `reports/metrics.json` (contains the full classification report)

### 3\. MinIO Console (http://localhost:9001)

Browse the MinIO console to see how DVC stores the versioned artifacts in your self-hosted object storage.

