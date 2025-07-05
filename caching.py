import sys
import os
import re
from datetime import datetime
from typing import List, Dict, Any, Tuple, Set
from google.cloud import storage
from utilities.caching.sqlite_caching import SQLiteBaseCache

# Optional: progress bar
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x: x

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
log_file_path = "error_log.txt"

def extract_mode_from_filename(filename: str) -> str:
    match = re.search(r"mode-([^-_/]+)", filename)
    return match.group(1) if match else None

def parse_gcs_path(gcs_path: str) -> Dict[str, Any]:
    path = gcs_path.replace("gs://", "")
    parts = path.split("/")
    if not gcs_path.endswith(".parquet") or len(parts) < 10:
        raise ValueError("Invalid or incomplete GCS path")

    filename = parts[-1]
    blkgrp_parts = parts[9].split("=")[-1].split("_")
    if len(blkgrp_parts) != 4:
        raise ValueError("Unexpected blkgrp format in path")

    return {
        "did": parts[2],
        "data_type": parts[3],
        "progname": parts[4].split("-")[-1],
        "step": parts[5].split("=")[-1],
        "lot": parts[6].split("=")[-1],
        "session": parts[7].split("=")[-1],
        "read_point": int(parts[8].split("=")[-1]),
        "blkgrp_prefix": blkgrp_parts[0],
        "blkgrp": int(blkgrp_parts[1]),
        "setup_mode": blkgrp_parts[2],
        "pb_setup": blkgrp_parts[3],
        "mode": extract_mode_from_filename(filename),
        "timestamp": datetime.now().isoformat()
    }

def list_gcs_parquet_files(bucket_name: str, prefix: str) -> List[str]:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    return [
        f"gs://{blob.bucket.name}/{blob.name}"
        for blob in bucket.list_blobs(prefix=prefix)
        if blob.name.endswith(".parquet")
    ]

def get_existing_keys(cache: SQLiteBaseCache) -> Set[Tuple]:
    rows = cache.query(
        f"SELECT did, data_type, progname, step, lot, session, read_point, blkgrp_prefix, blkgrp, setup_mode, pb_setup, mode FROM {cache.table_name}"
    )
    return set(rows)

def refresh_cache(cache: SQLiteBaseCache, gcs_paths: List[str]):
    existing_keys = get_existing_keys(cache)
    new_rows = []

    for path in tqdm(gcs_paths, desc="Processing GCS paths"):
        try:
            row = parse_gcs_path(path)
            key = (
                row["did"],
                row["data_type"],
                row["progname"],
                row["step"],
                row["lot"],
                row["session"],
                row["read_point"],
                row["blkgrp_prefix"],
                row["blkgrp"],
                row["setup_mode"],
                row["pb_setup"],
                row["mode"]
            )
            if key not in existing_keys:
                new_rows.append(row)
        except Exception as e:
            with open(log_file_path, "a") as log_file:
                log_file.write(f"Skipping path: {path} → {e}\n")

    if new_rows:
        print(f"Inserting {len(new_rows)} new rows...")
        cache.insert_many(new_rows)
    else:
        print("No new rows to insert.")

if __name__ == "__main__":
    # Config
    db_path = r"\\fsnvemaffs\nve_maf\cache\cache.db"
    # db_path = "/maf/nve_maf/cache/cache.db"  # For Unix

    table_name = "gcp_caching"
    schema = {
        "did": "TEXT", "data_type": "TEXT", "progname": "TEXT", "step": "TEXT",
        "lot": "TEXT", "session": "TEXT", "read_point": "INTEGER", "blkgrp_prefix": "TEXT",
        "blkgrp": "INTEGER", "setup_mode": "TEXT", "pb_setup": "TEXT",
        "mode": "TEXT", "timestamp": "DATETIME"
    }

    cache = SQLiteBaseCache(db_path, table_name, schema)

    bucket = "gdw-prod-data-maf-mhc"
    prefix = "gcDataDropboxParquet"

    gcs_paths = list_gcs_parquet_files(bucket, prefix)
    refresh_cache(cache, gcs_paths)
