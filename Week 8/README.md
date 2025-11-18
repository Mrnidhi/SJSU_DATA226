# HW8 - Pinecone Vector Search Pipeline

This project implements an end-to-end vector search pipeline using Airflow, Pinecone, and sentence transformers.

## Project Structure

```
.
├─ airflow/
│  ├─ dags/
│  │  └─ hw8_pinecone_pipeline.py
│  ├─ Dockerfile
│  ├─ requirements.txt
│  ├─ logs/                # git-ignored
│  └─ plugins/
├─ dbt/
│  └─ Dockerfile
├─ data/
│  └─ processed/hw8_input.jsonl
├─ .env                    # contains AIRFLOW_UID (no API keys!)
├─ docker-compose-celery.yaml
├─ screenshots/
└─ README.md
```

## Setup

1. Set up `.env` file with `AIRFLOW_UID` and `AIRFLOW_GID`
2. Configure Pinecone Airflow Variable `pinecone_cfg` with your API key and index settings
3. Build and start services: `docker compose -f docker-compose-celery.yaml up -d`

## Running the Pipeline

1. Open Airflow UI at http://localhost:8080
2. Enable the `hw8_pinecone_pipeline` DAG
3. Trigger the DAG to run all tasks:
   - `build_input_file`: Downloads and chunks text
   - `create_pinecone_index`: Creates Pinecone index (idempotent)
   - `embed_and_upsert`: Embeds chunks and upserts to Pinecone
   - `semantic_search`: Queries the index and returns results

## Requirements

- Docker and Docker Compose
- Pinecone API key
- Airflow Variables configured (see setup)

