-- One Postgres instance, several databases (handbook: reuse the MLflow Postgres for drift metrics, Langfuse, Airflow).
CREATE DATABASE langfuse;
CREATE DATABASE airflow;
CREATE DATABASE monitoring;
