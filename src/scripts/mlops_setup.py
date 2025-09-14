import subprocess
from pathlib import Path

from minio import Minio
from minio.error import S3Error

# Import our centralized settings loaders
from src.config.settings import app_config, env_config


# --- Main Setup Class ---

class MLOpsSetup:


    def __init__(self):
        print("🚀 Initializing MLOps Environment Setup...")

        self.root_dir = Path(__file__).resolve().parent.parent.parent
        self.paths = app_config.paths
        self.minio_config = app_config.minio
        self.minio_secrets = env_config

        print(f"   - Project Root: {self.root_dir}")
        print("   - ✅ Configuration loaded successfully.")

    def _create_directories(self):
        print("\n📂 Ensuring all necessary directories exist...")

        for path_str in vars(self.paths).values():
            dir_path = self.root_dir / Path(path_str).parent
            if not dir_path.exists():
                print(f"   - Creating directory: {dir_path}")
                dir_path.mkdir(parents=True, exist_ok=True)

    def _configure_minio(self):
        print(f"\n🪣 Checking for MinIO bucket '{self.minio_config.bucket_name}'...")
        try:
            client = Minio(
                self.minio_config.endpoint,
                access_key=self.minio_secrets.MINIO_ROOT_USER,
                secret_key=self.minio_secrets.MINIO_ROOT_PASSWORD,
                secure=False
            )
            if not client.bucket_exists(self.minio_config.bucket_name):
                print(f"   - Bucket not found. Creating bucket...")
                client.make_bucket(self.minio_config.bucket_name)
                print(f"   - ✅ Bucket '{self.minio_config.bucket_name}' created.")
            else:
                print(f"   - ✅ Bucket '{self.minio_config.bucket_name}' already exists.")
        except S3Error as e:
            print("\n❌ Could not connect to MinIO. Is the Docker container running?")
            print(f"   Error: {e}")
            exit(1)

    def _configure_dvc(self):
        print("\n🔄 Configuring DVC...")

        if not (self.root_dir / ".dvc").exists():
            print("   - .dvc directory not found. Initializing DVC...")
            subprocess.run(["dvc", "init"], cwd=self.root_dir, check=True)
        else:
            print("   - ✅ DVC already initialized.")

        print("   - 🔗 Configuring DVC remote 'storage'...")
        remote_url = f"s3://{self.minio_config.bucket_name}"
        endpoint_url = f"http://{self.minio_config.endpoint}"

        try:
            # Step 1: Add the remote (without extra config flags)
            subprocess.run(
                ["dvc", "remote", "add", "-d", "storage", remote_url, "--force"],
                cwd=self.root_dir, check=True, capture_output=True
            )
            # Step 2: Modify the remote with the endpoint URL
            subprocess.run(
                ["dvc", "remote", "modify", "storage", "endpointurl", endpoint_url],
                cwd=self.root_dir, check=True, capture_output=True
            )
            # Step 3: Modify the remote with the access key
            subprocess.run(
                ["dvc", "remote", "modify", "storage", "access_key_id", self.minio_secrets.MINIO_ROOT_USER],
                cwd=self.root_dir, check=True, capture_output=True
            )
            # Step 4: Modify the remote with the secret key
            subprocess.run(
                ["dvc", "remote", "modify", "storage", "secret_access_key", self.minio_secrets.MINIO_ROOT_PASSWORD],
                cwd=self.root_dir, check=True, capture_output=True
            )
            print("   - ✅ DVC remote 'storage' configured successfully.")
        except subprocess.CalledProcessError as e:
            print("\n❌ Failed to configure DVC remote.")
            print(f"   STDERR: {e.stderr.decode()}")
            exit(1)

    def run(self):
        self._create_directories()
        self._configure_minio()
        self._configure_dvc()
        print("\n🎉 Setup complete! Your MLOps environment is ready to use.")

if __name__ == "__main__":
    setup = MLOpsSetup()
    setup.run()