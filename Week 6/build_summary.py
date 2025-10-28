from datetime import datetime
from airflow import DAG
from airflow.providers.snowflake.operators.snowflake import SnowflakeOperator

DEFAULT_ARGS = {"owner": "data226", "retries": 0}

with DAG(
    dag_id="BuildSummary",
    description="ELT: join RAW tables into analytics.session_summary with duplicate check",
    default_args=DEFAULT_ARGS,
    schedule_interval=None,   # manual trigger
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=["elt", "snowflake", "analytics"],
) as dag:

    create_schema = SnowflakeOperator(
        task_id="create_schema",
        snowflake_conn_id="snowflake_default",
        sql="CREATE SCHEMA IF NOT EXISTS analytics;",
    )

    create_table = SnowflakeOperator(
        task_id="create_table",
        snowflake_conn_id="snowflake_default",
        sql="""
            CREATE TABLE IF NOT EXISTS analytics.session_summary (
              sessionId VARCHAR(32) PRIMARY KEY,
              userId INT,
              channel VARCHAR(32),
              ts TIMESTAMP,
              week_start DATE
            );
        """,
    )

    duplicate_guard = SnowflakeOperator(
        task_id="duplicate_guard",
        snowflake_conn_id="snowflake_default",
        sql="""
            SELECT COUNT(*) as duplicate_count FROM (
              SELECT sessionId, COUNT(*) c
              FROM raw.user_session_channel
              GROUP BY 1 HAVING c > 1
              UNION ALL
              SELECT sessionId, COUNT(*) c
              FROM raw.session_timestamp
              GROUP BY 1 HAVING c > 1
            );
        """,
    )

    upsert_summary = SnowflakeOperator(
        task_id="upsert_summary",
        snowflake_conn_id="snowflake_default",
        sql="""
            MERGE INTO analytics.session_summary t
            USING (
              SELECT
                u.sessionId,
                ANY_VALUE(u.userId) AS userId,
                ANY_VALUE(u.channel) AS channel,
                s.ts,
                DATE_TRUNC('WEEK', s.ts)::DATE AS week_start
              FROM raw.user_session_channel u
              JOIN raw.session_timestamp s
                ON u.sessionId = s.sessionId
              GROUP BY u.sessionId, s.ts
            ) src
            ON t.sessionId = src.sessionId
            WHEN MATCHED THEN UPDATE SET
              userId = src.userId,
              channel = src.channel,
              ts = src.ts,
              week_start = src.week_start
            WHEN NOT MATCHED THEN INSERT (sessionId, userId, channel, ts, week_start)
            VALUES (src.sessionId, src.userId, src.channel, src.ts, src.week_start);
        """,
    )

    create_schema >> create_table >> duplicate_guard >> upsert_summary
