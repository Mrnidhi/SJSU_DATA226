from __future__ import annotations

import json
import pathlib
import re
import uuid
import requests

from datetime import datetime
from airflow import DAG
from airflow.decorators import task
from airflow.models import Variable
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

INPUT_URL = "https://www.gutenberg.org/cache/epub/1513/pg1513.txt"
LOCAL_FALLBACK_TEXT = "This is a fallback paragraph about Airflow and Pinecone if the download fails."
CHUNK_SIZE = 1200
OVERLAP = 150
OUTPUT_JSONL = "/opt/airflow/data/processed/hw8_input.jsonl"

with DAG(
    dag_id="hw8_pinecone_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule=None,
    catchup=False,
    tags=["hw8", "pinecone", "embeddings"],
):

    @task
    def build_input_file() -> str:
        pathlib.Path("/opt/airflow/data/processed").mkdir(parents=True, exist_ok=True)

        try:
            r = requests.get(INPUT_URL, timeout=20)
            r.raise_for_status()
            text = r.text
            source = INPUT_URL
        except Exception:
            text = LOCAL_FALLBACK_TEXT
            source = "local_fallback"

        text = re.sub(r"\s+", " ", text).strip()

        chunks = []
        i = 0
        while i < len(text):
            chunk = text[i : i + CHUNK_SIZE]
            if len(chunk) < CHUNK_SIZE * 0.4:
                break
            chunks.append(chunk)
            i += CHUNK_SIZE - OVERLAP

        count = 0
        with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
            for j, chunk in enumerate(chunks):
                rec = {
                    "id": str(uuid.uuid4()),
                    "text": chunk,
                    "metadata": {"source": source, "chunk": j},
                }
                f.write(json.dumps(rec) + "\n")
                count += 1

        print(f"Wrote {count} records to {OUTPUT_JSONL}")
        return OUTPUT_JSONL

    @task
    def create_pinecone_index():
        cfg = Variable.get("pinecone_cfg", deserialize_json=True)
        pc = Pinecone(api_key=cfg["api_key"])
        index_name = cfg["index_name"]

        existing = [i["name"] for i in pc.list_indexes()]
        if index_name not in existing:
            pc.create_index(
                name=index_name,
                dimension=cfg["dimension"],
                metric=cfg.get("metric", "cosine"),
                spec=ServerlessSpec(
                    cloud=cfg.get("cloud", "aws"),
                    region=cfg.get("region", "us-east-1")
                ),
            )
            print(f"Created new index: {index_name}")
        else:
            print(f"Index already exists: {index_name}")
        return index_name

    @task
    def embed_and_upsert(input_jsonl: str):
        cfg = Variable.get("pinecone_cfg", deserialize_json=True)
        pc = Pinecone(api_key=cfg["api_key"])
        index = pc.Index(cfg["index_name"])

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        records = []
        with open(input_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                records.append(json.loads(line))

        batch_size = 64
        total = 0
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            texts = [r["text"] for r in batch]
            ids = [r["id"] for r in batch]
            metas = [r.get("metadata", {}) for r in batch]

            embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            vectors = [
                {"id": ids[j], "values": embeddings[j].tolist(), "metadata": metas[j]}
                for j in range(len(batch))
            ]
            index.upsert(vectors=vectors)
            total += len(batch)
            print(f"Upserted {total}/{len(records)}")

        print(f"Finished upserting {total} vectors into '{cfg['index_name']}'")
        return total

    @task
    def semantic_search(query: str = "Where do Romeo and Juliet first meet?"):
        cfg = Variable.get("pinecone_cfg", deserialize_json=True)
        pc = Pinecone(api_key=cfg["api_key"])
        index = pc.Index(cfg["index_name"])

        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        query_vector = model.encode([query], convert_to_numpy=True)[0].tolist()

        top_k = int(cfg.get("top_k", 5))
        results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)

        print(f"Query: {query}  | top_k={top_k}")
        matches = results.get("matches", []) if isinstance(results, dict) else getattr(results, "matches", [])
        for i, match in enumerate(matches, 1):
            match_id = match.get("id") if isinstance(match, dict) else match.id
            score = match.get("score") if isinstance(match, dict) else match.score
            metadata = match.get("metadata") if isinstance(match, dict) else getattr(match, "metadata", {})
            print(f"{i}) id={match_id}  score={score:.4f}  meta={metadata}")
        print(f"Total matches returned: {len(matches)}")
        return True

    input_path = build_input_file()
    index_name = create_pinecone_index()
    upserted_count = embed_and_upsert(input_path)
    search_done = semantic_search()

    index_name.set_upstream(input_path)
    upserted_count.set_upstream(index_name)
    search_done.set_upstream(upserted_count)

