#!/bin/bash
set -e

ROOT="$(dirname "$0")/.."
ENVFILE="$ROOT/minio/.env"

if [ ! -f "$ENVFILE" ]; then
  echo "❌ Cannot find $ENVFILE"
  exit 1
fi

export $(grep -v '^#' "$ENVFILE" | xargs)

cd "$ROOT"

# Initialize DVC if not already present
if [ ! -d ".dvc" ]; then
  echo "🔄 Initializing DVC..."
  dvc init
fi

# Configure DVC remote
dvc remote add -d minio s3://mlflow-dvc-bucket/models --force
dvc remote modify minio endpointurl http://localhost:9000
dvc remote modify minio access_key_id "$MINIO_ROOT_USER"
dvc remote modify minio secret_access_key "$MINIO_ROOT_PASSWORD"

echo "✅ DVC initialized and remote 'minio' configured at http://localhost:9000"
