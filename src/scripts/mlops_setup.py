#src.scripts.mlops_setup.py
import subprocess
from pathlib import Path

from minio import Minio
from minio.error import S3Error

from src.config.settings import app_config, env_config


class MLOpsSetup:

    def __init__(self):
        print("🚀 Initializing MLOps Environment Setup...")
        self.root_dir = Path(__file__).resolve().parent.parent.parent
        self.paths = app_config.paths
        self.minio_config = app_config.minio
        self.minio_secrets = env_config
        print(f"   - Project Root: {self.root_dir}")
        print("   - ✅ Configuration loaded successfully.")

    def _run_command(self, command: list):
        try:
            print(f"   - Executing: {' '.join(command)}")
            subprocess.run(command, check=True, text=True, cwd=self.root_dir)
        except subprocess.CalledProcessError as e:
            print(f"❌ Command failed: {' '.join(command)}")
            print(f"   Error: {e}")
            exit(1)

    def _create_directories(self):
        print(".n📂 Ensuring all necessary directories exist...")
        for path_str in vars(self.paths).values():
            dir_path = self.root_dir / Path(path_str).parent
            if not dir_path.exists():
                print(f"   - Creating directory: {dir_path}")
                dir_path.mkdir(parents=True, exist_ok=True)

    def _configure_minio(self):
        print(f".n🪣 Checking for MinIO bucket '{self.minio_config.bucket_name}'...")
        try:
            client = Minio(
                self.minio_config.endpoint,
                access_key=self.minio_secrets.MINIO_ROOT_USER,
                secret_key=self.minio_secrets.MINIO_ROOT_PASSWORD,
                secure=False
            )
            if not client.bucket_exists(self.minio_config.bucket_name):
                client.make_bucket(self.minio_config.bucket_name)
                print(f"   - ✅ Bucket '{self.minio_config.bucket_name}' created.")
            else:
                print(f"   - ✅ Bucket '{self.minio_config.bucket_name}' already exists.")
        except S3Error as e:
            print(".n❌ Could not connect to MinIO. Is the Docker container running?")
            print(f"   Error: {e}")
            exit(1)

    def _configure_dvc(self):
        print(".n🔄 Configuring DVC...")
        if not (self.root_dir / ".dvc").exists():
            self._run_command(["dvc", "init"])
        else:
            print("   - ✅ DVC already initialized.")

        print("   - 🔗 Configuring DVC remote 'storage'...")
        remote_url = f"s3://{self.minio_config.bucket_name}"
        endpoint_url = f"http://{self.minio_config.endpoint}"

        self._run_command(["dvc", "remote", "add", "-d", "storage", remote_url, "--force"])
        self._run_command(["dvc", "remote", "modify", "storage", "endpointurl", endpoint_url])
        self._run_command(["dvc", "remote", "modify", "storage", "access_key_id", self.minio_secrets.MINIO_ROOT_USER])
        self._run_command(
            ["dvc", "remote", "modify", "storage", "secret_access_key", self.minio_secrets.MINIO_ROOT_PASSWORD])
        print("   - ✅ DVC remote 'storage' configured successfully.")

    def _pull_data(self):
        print(".n⏬ Pulling latest data and models from DVC remote...")
        self._run_command(["dvc", "pull"])

    def run(self):
        self._create_directories()
        self._configure_minio()
        self._configure_dvc()
        self._pull_data()  # ✅ Add the pull step to the end
        print(".n🎉 Setup and bootstrap complete! Your environment is ready.")


if __name__ == "__main__":
    setup = MLOpsSetup()
    setup.run()