import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import re
from datetime import datetime
from google.cloud import storage
from typing import List, Dict, Any, Tuple, Set
from utilities.caching.sqlite_caching import SQLiteBaseCache

log_file_path = "error_log.txt"

def extract_mode_from_filename(filename: str) -> str:
    match = re.search(r"mode-([^-_/]+)", filename)
    return match.group(1) if match else None

def parse_gcs_path(gcs_path: str) -> Dict[str, Any]:
    path = gcs_path.replace("gs://", "")
    parts = path.split("/")
    if not path.endswith(".parquet") or len(parts) < 9:
        raise ValueError("Invalid or incomplete GCS path")
 
    filename = parts[-1]

    try:
        return {
            "did": parts[2],
            "data_type": parts[3],
            "progname": parts[4].split("-")[-1],
            "step": parts[5].split("=")[-1],
            "lot": parts[6].split("=")[-1],
            "session": parts[7].split("=")[-1],
            "read_point": int(parts[8].split("=")[-1]),
            "blkgrp_prefix": parts[9].split("=")[-1].split("_")[0], 
            "blkgrp": int(parts[9].split("=")[-1].split("_")[1]),
            "setup_mode": parts[9].split("=")[-1].split("_")[2],
            "pb_setup": parts[9].split("=")[-1].split("_")[3],
            "mode": extract_mode_from_filename(filename),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        print(f"Failed due to Exception: {e}")
        print(path)

def list_gcs_parquet_files(bucket_name: str, prefix: str) -> List[str]:
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    return [
f"gs://{blob.bucket.name}/{blob.name}"
        for blob in bucket.list_blobs(prefix=prefix)
if blob.name.endswith(".parquet")
    ]
 
def get_existing_keys(cache: SQLiteBaseCache) -> Set[Tuple[str, str, str, str]]:
    rows = cache.query(f"SELECT DISTINCT did,data_type,progname,step,lot,session,read_point,blkgrp,setup_mode,pb_setup,mode FROM {cache.table_name}")
    return {(did,data_type,progname,step,lot,session,read_point,blkgrp,setup_mode,pb_setup,mode) for did,data_type,progname,step,lot,session,read_point,blkgrp,setup_mode,pb_setup,mode in rows}

log_file_path = "error_log.txt"

def refresh_cache(cache: SQLiteBaseCache, gcs_paths: List[str]):
    existing_keys = get_existing_keys(cache)
    new_rows = []

    # print(existing_keys)
 
    for path in gcs_paths:
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
    # Setup
    db_path = r"\\fsnvemaffs\nve_maf\cache\cache.db"
    # if unix
    # db_path = "/maf/nve_maf/cache/cache.db"
    table_name = "gcp_caching"
    schema = {
            "did": "TEXT",
            "data_type": "TEXT",
            "progname": "TEXT",
            "step": "TEXT",
            "lot": "TEXT",
            "session": "TEXT",
            "read_point": "INTEGER",
            "blkgrp_prefix": "TEXT",
            "blkgrp": "INTEGER",
            "setup_mode": "TEXT",
            "pb_setup": "TEXT",
            "mode": "TEXT",
            "timestamp": "DATETIME"
        }

    cache = SQLiteBaseCache(db_path, table_name, schema)

    # Modify as needed:
    bucket = "gdw-prod-data-maf-mhc"
    prefix = "gcDataDropboxParquet"

    gcs_paths = list_gcs_parquet_files(bucket, prefix)

    # print(len(gcs_paths))
    refresh_cache(cache, gcs_paths)
