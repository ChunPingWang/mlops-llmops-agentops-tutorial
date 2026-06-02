# Dagster webserver + daemon image with project deps baked in. The official
# dagster/dagster-webserver image lacks the project's pip deps (pandas,
# scikit-learn, mlflow, dotenv, ...).
FROM python:3.11-slim

RUN pip install --no-cache-dir \
    "dagster>=1.9.0" \
    "dagster-webserver>=1.9.0" \
    "dagster-postgres>=0.25.0" \
    "pandas>=2.2.0" \
    "numpy>=1.26.0" \
    "scikit-learn>=1.5.0" \
    "mlflow>=2.17.0" \
    "python-dotenv>=1.0.0" \
    "psycopg2-binary>=2.9.9"

ENV DAGSTER_HOME=/opt/dagster/dagster_home
RUN mkdir -p $DAGSTER_HOME /opt/dagster/app

WORKDIR /opt/dagster/app
EXPOSE 3000
