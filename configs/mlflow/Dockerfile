# Custom MLflow image that ships with psycopg2-binary so --backend-store-uri
# postgresql://... works out of the box. The ghcr.io/mlflow image does not.
FROM ghcr.io/mlflow/mlflow:latest

RUN pip install --no-cache-dir psycopg2-binary>=2.9.9 boto3>=1.34.0
