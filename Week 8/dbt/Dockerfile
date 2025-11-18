FROM python:3.11-slim

# system deps (optional but handy for dbt & Snowflake)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git build-essential libpq-dev curl && rm -rf /var/lib/apt/lists/*

# dbt core + Snowflake adapter (per class slides using dbt Core) 
RUN pip install --no-cache-dir "dbt-snowflake==1.8.*"

WORKDIR /dbt
# Default command keeps the container alive for execs from Airflow (bash, etc.)
CMD ["tail","-f","/dev/null"]

