from datetime import datetime
from airflow import DAG
from airflow.providers.snowflake.operators.snowflake import SnowflakeOperator

DEFAULT_ARGS = {"owner": "data226", "retries": 0}

with DAG(
    dag_id="SessionToSnowflake",
    description="ETL: Load raw.user_session_channel and raw.session_timestamp into Snowflake",
    default_args=DEFAULT_ARGS,
    schedule_interval=None,   # manual trigger
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["etl", "snowflake", "raw"],
) as dag:

    create_tables = SnowflakeOperator(
        task_id="create_tables",
        snowflake_conn_id="snowflake_default",
        sql=[
            """
            CREATE TABLE IF NOT EXISTS raw.user_session_channel (
                userId INT NOT NULL,
                sessionId VARCHAR(32) PRIMARY KEY,
                channel VARCHAR(32) DEFAULT 'direct'
            );
            """,
            """
            CREATE TABLE IF NOT EXISTS raw.session_timestamp (
                sessionId VARCHAR(32) PRIMARY KEY,
                ts TIMESTAMP
            );
            """,
        ],
    )

    create_stage = SnowflakeOperator(
        task_id="create_stage",
        snowflake_conn_id="snowflake_default",
        sql="""
            CREATE OR REPLACE STAGE raw.blob_stage
            URL='s3://s3-geospatial/readonly/'
            FILE_FORMAT=(TYPE=CSV SKIP_HEADER=1 FIELD_OPTIONALLY_ENCLOSED_BY='"');
        """,
    )

    copy_user_session_channel = SnowflakeOperator(
        task_id="copy_user_session_channel",
        snowflake_conn_id="snowflake_default",
        sql="COPY INTO raw.user_session_channel FROM @raw.blob_stage/user_session_channel.csv;",
    )

    copy_session_timestamp = SnowflakeOperator(
        task_id="copy_session_timestamp",
        snowflake_conn_id="snowflake_default",
        sql="COPY INTO raw.session_timestamp FROM @raw.blob_stage/session_timestamp.csv;",
    )

    create_tables >> create_stage >> [copy_user_session_channel, copy_session_timestamp]
