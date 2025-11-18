from datetime import datetime
from airflow import DAG
from airflow.decorators import task
from airflow.models import Variable

with DAG(
    dag_id="hw8_pinecone_setup_check",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["hw8", "pinecone"]
):
    @task
    def show_config():
        cfg = Variable.get("pinecone_cfg", deserialize_json=True)
        # Avoid printing secrets:
        safe = {**cfg, "api_key": "***redacted***"}
        print("Pinecone config:", safe)

    show_config()
