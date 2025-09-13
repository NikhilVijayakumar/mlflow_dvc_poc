@echo off
setlocal enabledelayedexpansion

REM Get project root (one folder up from scripts)
set ROOT=%~dp0..
set ENVFILE=%ROOT%\minio\.env

if not exist "%ENVFILE%" (
    echo ❌ Cannot find %ENVFILE%
    exit /b 1
)

REM Load environment variables from minio\.env
for /f "tokens=1,2 delims==" %%a in (%ENVFILE%) do (
    set %%a=%%b
)

REM Go to project root
cd %ROOT%

REM Initialize DVC if not already done
if not exist ".dvc" (
    echo 🔄 Initializing DVC...
    dvc init
)

REM Configure DVC remote
dvc remote add -d minio s3://mlflow-dvc-bucket/models --force
dvc remote modify minio endpointurl http://localhost:9000
dvc remote modify minio access_key_id %MINIO_ROOT_USER%
dvc remote modify minio secret_access_key %MINIO_ROOT_PASSWORD%

echo ✅ DVC initialized and remote 'minio' configured at http://localhost:9000
