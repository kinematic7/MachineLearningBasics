import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import json
import re
import random
from typing import List
import psycopg2
from sentence_transformers import SentenceTransformer
from pgvector.psycopg2 import register_vector

# ----------------------------
# Initialize embedding model
# ----------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')

# ----------------------------
# Database configurations
# ----------------------------
DB_CONFIGS = {
    "local": {
        "host": "localhost",
        "database": "Rubaiyat",
        "user": "postgres",
        "password": "com156sA",
        "options": "-c search_path=Rubaiyat,public"
    },
    "prod": {
        "host": "rubaiyat.chmigquya572.us-east-1.rds.amazonaws.com",
        "database": "Rubaiyat",
        "user": "mnislam",
        "password": "com156sA",
        "options": "-c search_path=Rubaiyat,public"
    }
}

DATA_DIR = "vector_store_data"
os.makedirs(DATA_DIR, exist_ok=True)

# ----------------------------
# JSON ingestion
# ----------------------------
def ingest_json_to_postgres(file_path: str, env: str = "prod", batch_size: int = 100) -> int:
    if env not in DB_CONFIGS:
        raise ValueError(f"Invalid environment: {env}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    filename = os.path.basename(file_path)
    config = DB_CONFIGS[env]

    conn = psycopg2.connect(**config)

    try:
        with conn.cursor() as cur:
            # CRITICAL: Set search path so the DB can find the vector type in 'public'
            cur.execute('SET search_path TO "Rubaiyat", public;')
            
            # Register vector type AFTER setting the search path
            register_vector(conn)

            records_count = 0
            for i in range(0, len(data), batch_size):
                batch = data[i:i+batch_size]
                for rec in batch:
                    text = rec.get("text", "")
                    if not text:
                        continue

                    embedding = model.encode(text).tolist()
                    cur.execute(
                        'INSERT INTO "document_embeddings" (filename, content, embedding) VALUES (%s, %s, %s)',
                        (filename, text, embedding)
                    )
                    records_count += 1

            conn.commit()

        print(f"Successfully ingested {records_count} records into '{env}' database.")
        return records_count

    except Exception as e:
        conn.rollback()
        print(f"Database Error: {e}")
        raise e
    finally:
        conn.close()


# ----------------------------
# Semantic search
# ----------------------------
def semantic_search_postgres(tags: List[str], top_k: int = 50, top_shuffle: int = 50, env: str = "prod") -> List[dict]:
    if not tags:
        return []

    config = DB_CONFIGS[env]
    conn = psycopg2.connect(**config)

    all_results = []
    try:
        with conn.cursor() as cur:
            cur.execute('SET search_path TO "Rubaiyat", public;')
            register_vector(conn)

            for tag in tags:
                tag_text = tag.strip()
                if not tag_text:
                    continue

                tag_embedding = model.encode(tag_text).tolist()
                cur.execute("""
                    SELECT content, 1 - (embedding <=> %s::vector) AS similarity
                    FROM "document_embeddings"
                    ORDER BY embedding <=> %s::vector ASC
                    LIMIT %s;
                """, (tag_embedding, tag_embedding, top_k))

                rows = cur.fetchall()
                for row in rows:
                    all_results.append({
                        "tag": tag_text,
                        "content": row[0],
                        "similarity": round(row[1], 4)
                    })

    finally:
        conn.close()

    random.shuffle(all_results)
    return all_results[:top_shuffle]


if __name__ == "__main__":
    # Test block
    query = "Trip to Bangladesh with [Alice] and [Bob]"
    tags = re.findall(r"\[([^\]]+)\]", query)

    # ingest_json_to_postgres("C:/tmp/Test.json", env="prod")
    results = semantic_search_postgres(tags, top_k=3)
    print("Top matches:", results)