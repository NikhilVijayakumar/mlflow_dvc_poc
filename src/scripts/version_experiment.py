# src/scripts/version_experiment.py
import subprocess
from datetime import datetime

from src.config.settings import app_config


class ExperimentVersioner:
    """
    A class to run the DVC pipeline and then version the results
    using Git and DVC. This creates a complete, atomic experiment run.
    """

    def __init__(self):
        print("🚀 Initializing Experiment Versioner...")
        self.paths = app_config.paths
        self.mlflow_config = app_config.mlflow

    def _run_command(self, command: list):
        """Helper function to run a command and handle errors."""
        try:
            print(f"   - Executing: {' '.join(command)}")
            subprocess.run(command, check=True, text=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ Command failed: {' '.join(command)}")
            print(f"   Error: {e}")
            exit(1)

    def run(self):
        """Executes the full repro, git commit, and dvc push workflow."""

        # 1. Run the DVC pipeline
        print("\n[Step 1/4] Reproducing DVC pipeline...")
        self._run_command(["dvc", "repro"])

        # 2. Add files to Git staging
        print("\n[Step 2/4] Staging results for Git...")
        files_to_add = [
            "dvc.lock"
        ]
        self._run_command(["git", "add"] + files_to_add)

        # 3. Commit the changes
        print("\n[Step 3/4] Committing experiment results...")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        commit_message = f"{self.mlflow_config.commit_message_template} ({timestamp})"
        self._run_command(["git", "commit", "-m", commit_message])
        print(f"   - ✅ Committed with message: '{commit_message}'")

        # 4. Push DVC artifacts to remote storage
        print("\n[Step 4/4] Pushing DVC artifacts to remote storage...")
        self._run_command(["dvc", "push"])

        print("\n🏁 Experiment versioning complete!")


if __name__ == "__main__":
    versioner = ExperimentVersioner()
    versioner.run()