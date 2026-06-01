#!/bin/bash
set -e

# Create databases and users for all services that need their own schema.
# This script is executed once by the PostgreSQL entrypoint on first start.

DATABASES=("litellm" "langfuse" "mlflow" "dagster")

for db in "${DATABASES[@]}"; do
  echo "Creating user and database: $db"
  psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE USER $db WITH PASSWORD '${POSTGRES_PASSWORD}';
    CREATE DATABASE $db;
    GRANT ALL PRIVILEGES ON DATABASE $db TO $db;
EOSQL

  # Grant schema-level privileges so the user can create tables
  psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$db" <<-EOSQL
    GRANT ALL ON SCHEMA public TO $db;
EOSQL
done

echo "All databases and users created successfully."
