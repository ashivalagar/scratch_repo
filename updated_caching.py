import sys
import os
import re
import logging
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set, Optional, Iterator
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache, wraps
import threading
from contextlib import contextmanager

from tqdm import tqdm
from google.cloud import storage
from google.api_core import exceptions as gcs_exceptions

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utilities.caching.sqlite_caching import SQLiteBaseCache


@dataclass
class GCSPathData:
    """Data structure for parsed GCS path information."""
    did: str
    data_type: str
    progname: str
    step: str
    lot: str
    session: str
    read_point: int
    blkgrp_prefix: str
    blkgrp: int
    setup_mode: str
    pb_setup: str
    mode: Optional[str]
    timestamp: str
    original_path: str = ""
    
    def to_cache_key(self) -> Tuple:
        """Generate cache key tuple for deduplication."""
        return (
            self.did, self.data_type, self.progname, self.step,
            self.lot, self.session, self.read_point, self.blkgrp_prefix,
            self.blkgrp, self.setup_mode, self.pb_setup, self.mode
        )


class EnhancedLogger:
    """Enhanced logging with structured error caching and multiple output formats."""
    
    def __init__(self, log_dir: str = "logs", log_level: str = "INFO"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.log_level = getattr(logging, log_level.upper())
        
        # Setup main logger
        self.logger = logging.getLogger("gcs_caching")
        self.logger.setLevel(self.log_level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        simple_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # File handlers
        self._setup_file_handlers(detailed_formatter)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        self.logger.addHandler(console_handler)
        
        # Error cache for analytics
        self.error_cache = {}
        self.error_lock = threading.Lock()
    
    def _setup_file_handlers(self, formatter):
        """Setup file handlers for different log levels."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Main log file
        main_handler = logging.FileHandler(self.log_dir / f"gcs_caching_{timestamp}.log")
        main_handler.setLevel(self.log_level)
        main_handler.setFormatter(formatter)
        self.logger.addHandler(main_handler)
        
        # Error-only log file
        error_handler = logging.FileHandler(self.log_dir / f"errors_{timestamp}.log")
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        self.logger.addHandler(error_handler)
        
        # JSON structured log for programmatic analysis
        json_handler = logging.FileHandler(self.log_dir / f"structured_{timestamp}.jsonl")
        json_handler.setLevel(logging.WARNING)
        json_handler.setFormatter(logging.Formatter('%(message)s'))
        self.json_handler = json_handler
        self.logger.addHandler(json_handler)
    
    def log_structured_error(self, error_type: str, path: str, error: Exception, context: Dict = None):
        """Log structured error for analytics and caching."""
        error_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error_type": error_type,
            "path": path,
            "error_message": str(error),
            "error_class": error.__class__.__name__,
            "context": context or {}
        }
        
        # Cache error for analytics
        with self.error_lock:
            error_key = f"{error_type}:{error.__class__.__name__}"
            if error_key not in self.error_cache:
                self.error_cache[error_key] = {"count": 0, "examples": [], "first_seen": error_data["timestamp"]}
            
            self.error_cache[error_key]["count"] += 1
            if len(self.error_cache[error_key]["examples"]) < 5:
                self.error_cache[error_key]["examples"].append(path)
        
        # Log to structured file
        self.json_handler.emit(logging.LogRecord(
            name="structured", level=logging.ERROR, pathname="", lineno=0,
            msg=json.dumps(error_data), args=(), exc_info=None
        ))
        
        # Log to main logger
        self.logger.error(f"{error_type} - {path}: {error}")
    
    def get_error_summary(self) -> Dict:
        """Get summary of cached errors for reporting."""
        with self.error_lock:
            return dict(self.error_cache)
    
    def info(self, msg: str): self.logger.info(msg)
    def warning(self, msg: str): self.logger.warning(msg)
    def error(self, msg: str): self.logger.error(msg)
    def debug(self, msg: str): self.logger.debug(msg)


class GCSPathProcessor:
    """Enhanced GCS path processing with caching and error handling."""
    
    def __init__(self, logger: EnhancedLogger):
        self.logger = logger
        self._mode_regex = re.compile(r"mode-([^-_/]+)")
        self._path_cache = {}
        self._cache_lock = threading.Lock()
    
    @lru_cache(maxsize=1000)
    def extract_mode_from_filename(self, filename: str) -> Optional[str]:
        """Extract mode from filename with caching."""
        match = self._mode_regex.search(filename)
        return match.group(1) if match else None
    
    def parse_gcs_path(self, gcs_path: str) -> Optional[GCSPathData]:
        """Parse GCS path with enhanced error handling and validation."""
        try:
            # Check cache first
            with self._cache_lock:
                if gcs_path in self._path_cache:
                    return self._path_cache[gcs_path]
            
            if not gcs_path.startswith("gs://"):
                raise ValueError("Path must start with gs://")
            
            path = gcs_path[5:]  # Remove "gs://"
            parts = path.split("/")
            
            if not gcs_path.endswith(".parquet"):
                raise ValueError("Path must end with .parquet")
            
            if len(parts) < 10:
                raise ValueError(f"Path has insufficient parts: {len(parts)} < 10")
            
            filename = parts[-1]
            
            # Validate and parse blkgrp
            if not parts[9].startswith("blkgrp="):
                raise ValueError("Expected blkgrp= in 10th path component")
            
            blkgrp_value = parts[9].split("=", 1)[-1]
            blkgrp_parts = blkgrp_value.split("_")
            
            if len(blkgrp_parts) != 4:
                raise ValueError(f"Unexpected blkgrp format: expected 4 parts, got {len(blkgrp_parts)}")
            
            # Parse and validate numeric fields
            try:
                read_point = int(parts[8].split("=")[-1])
                blkgrp = int(blkgrp_parts[1])
            except ValueError as e:
                raise ValueError(f"Invalid numeric field: {e}")
            
            # Create data object
            data = GCSPathData(
                did=parts[2],
                data_type=parts[3],
                progname=parts[4].split("-")[-1],
                step=parts[5].split("=")[-1],
                lot=parts[6].split("=")[-1],
                session=parts[7].split("=")[-1],
                read_point=read_point,
                blkgrp_prefix=blkgrp_parts[0],
                blkgrp=blkgrp,
                setup_mode=blkgrp_parts[2],
                pb_setup=blkgrp_parts[3],
                mode=self.extract_mode_from_filename(filename),
                timestamp=datetime.now(timezone.utc).isoformat(),
                original_path=gcs_path
            )
            
            # Cache the result
            with self._cache_lock:
                self._path_cache[gcs_path] = data
            
            return data
            
        except Exception as e:
            self.logger.log_structured_error("PATH_PARSE_ERROR", gcs_path, e, {
                "path_parts_count": len(parts) if 'parts' in locals() else 0
            })
            return None


class EnhancedGCSCacheManager:
    """Enhanced GCS cache manager with improved efficiency and error handling."""
    
    def __init__(self, db_path: str, table_name: str, schema: Dict[str, str], 
                 logger: EnhancedLogger, max_workers: int = 4):
        self.cache = SQLiteBaseCache(db_path, table_name, schema)
        self.logger = logger
        self.processor = GCSPathProcessor(logger)
        self.max_workers = max_workers
        self._stats = {
            "processed": 0,
            "successful": 0,
            "failed": 0,
            "skipped_existing": 0,
            "new_insertions": 0
        }
    
    @contextmanager
    def gcs_client(self):
        """Context manager for GCS client with proper cleanup."""
        client = None
        try:
            client = storage.Client()
            yield client
        except Exception as e:
            self.logger.log_structured_error("GCS_CLIENT_ERROR", "client_creation", e)
            raise
        finally:
            if client:
                # Cleanup if needed
                pass
    
    def list_gcs_parquet_files(self, bucket_name: str, prefix: str, 
                              batch_size: int = 1000) -> Iterator[List[str]]:
        """List GCS parquet files in batches for memory efficiency."""
        try:
            with self.gcs_client() as client:
                bucket = client.bucket(bucket_name)
                batch = []
                
                self.logger.info(f"Starting to list files from gs://{bucket_name}/{prefix}")
                
                for blob in bucket.list_blobs(prefix=prefix):
                    if blob.name.endswith(".parquet"):
                        batch.append(f"gs://{blob.bucket.name}/{blob.name}")
                        
                        if len(batch) >= batch_size:
                            yield batch
                            batch = []
                
                # Yield remaining files
                if batch:
                    yield batch
                    
        except gcs_exceptions.NotFound:
            self.logger.log_structured_error("GCS_BUCKET_ERROR", bucket_name, 
                                           Exception("Bucket not found"), {"prefix": prefix})
        except Exception as e:
            self.logger.log_structured_error("GCS_LIST_ERROR", f"{bucket_name}/{prefix}", e)
            raise
    
    def get_existing_keys(self) -> Set[Tuple]:
        """Get existing cache keys with error handling."""
        try:
            self.logger.info("Fetching existing cache keys...")
            rows = self.cache.query(
                f"SELECT did, data_type, progname, step, lot, session, read_point, "
                f"blkgrp_prefix, blkgrp, setup_mode, pb_setup, mode FROM {self.cache.table_name}"
            )
            self.logger.info(f"Found {len(rows)} existing records")
            return set(rows)
        except Exception as e:
            self.logger.log_structured_error("CACHE_QUERY_ERROR", "get_existing_keys", e)
            return set()
    
    def process_paths_batch(self, paths: List[str], existing_keys: Set[Tuple]) -> List[Dict[str, Any]]:
        """Process a batch of paths with parallel processing."""
        new_rows = []
        
        def process_single_path(path: str) -> Optional[Dict[str, Any]]:
            self._stats["processed"] += 1
            
            try:
                parsed_data = self.processor.parse_gcs_path(path)
                if not parsed_data:
                    self._stats["failed"] += 1
                    return None
                
                cache_key = parsed_data.to_cache_key()
                if cache_key in existing_keys:
                    self._stats["skipped_existing"] += 1
                    return None
                
                self._stats["successful"] += 1
                return asdict(parsed_data)
                
            except Exception as e:
                self._stats["failed"] += 1
                self.logger.log_structured_error("PATH_PROCESS_ERROR", path, e)
                return None
        
        # Process paths in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_path = {executor.submit(process_single_path, path): path 
                            for path in paths}
            
            for future in as_completed(future_to_path):
                result = future.result()
                if result:
                    new_rows.append(result)
        
        return new_rows
    
    def refresh_cache(self, bucket_name: str, prefix: str, batch_size: int = 1000):
        """Enhanced cache refresh with batching and improved error handling."""
        start_time = time.time()
        self.logger.info(f"Starting cache refresh for gs://{bucket_name}/{prefix}")
        
        try:
            # Get existing keys once
            existing_keys = self.get_existing_keys()
            
            # Process files in batches
            total_new_rows = 0
            
            for batch_paths in self.list_gcs_parquet_files(bucket_name, prefix, batch_size):
                self.logger.info(f"Processing batch of {len(batch_paths)} files...")
                
                with tqdm(desc=f"Processing batch", total=len(batch_paths)) as pbar:
                    new_rows = self.process_paths_batch(batch_paths, existing_keys)
                    pbar.update(len(batch_paths))
                
                # Insert new rows if any
                if new_rows:
                    try:
                        self.cache.insert_many(new_rows)
                        total_new_rows += len(new_rows)
                        self._stats["new_insertions"] += len(new_rows)
                        self.logger.info(f"Inserted {len(new_rows)} new rows in batch")
                    except Exception as e:
                        self.logger.log_structured_error("CACHE_INSERT_ERROR", "batch_insert", e, {
                            "batch_size": len(new_rows)
                        })
            
            # Final statistics
            elapsed_time = time.time() - start_time
            self.logger.info(f"Cache refresh completed in {elapsed_time:.2f}s")
            self.logger.info(f"Statistics: {self._stats}")
            
            if total_new_rows == 0:
                self.logger.info("No new rows to insert - cache is up to date")
            else:
                self.logger.info(f"Successfully inserted {total_new_rows} new rows total")
                
        except Exception as e:
            self.logger.log_structured_error("CACHE_REFRESH_ERROR", f"{bucket_name}/{prefix}", e)
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            **self._stats,
            "error_summary": self.logger.get_error_summary()
        }


def main():
    """Main execution function with comprehensive error handling."""
    # Configuration
    config = {
        "db_path": r"\\fsnvemaffs\nve_maf\cache\cache.db",
        # "db_path": "/maf/nve_maf/cache/cache.db",  # For Unix
        "table_name": "gcp_caching",
        "schema": {
            "did": "TEXT", "data_type": "TEXT", "progname": "TEXT", "step": "TEXT",
            "lot": "TEXT", "session": "TEXT", "read_point": "INTEGER", "blkgrp_prefix": "TEXT",
            "blkgrp": "INTEGER", "setup_mode": "TEXT", "pb_setup": "TEXT",
            "mode": "TEXT", "timestamp": "DATETIME", "original_path": "TEXT"
        },
        "bucket": "gdw-prod-data-maf-mhc",
        "prefix": "gcDataDropboxParquet/*/dyn_read/mv2-EBQWPST",
        "batch_size": 1000,
        "max_workers": 4,
        "log_level": "INFO"
    }
    
    # Initialize logger
    logger = EnhancedLogger(log_level=config["log_level"])
    
    try:
        logger.info("=== GCS Cache Refresh Started ===")
        logger.info(f"Configuration: {json.dumps({k: v for k, v in config.items() if k != 'schema'}, indent=2)}")
        
        # Initialize cache manager
        cache_manager = EnhancedGCSCacheManager(
            config["db_path"], 
            config["table_name"], 
            config["schema"],
            logger,
            config["max_workers"]
        )
        
        # Refresh cache
        cache_manager.refresh_cache(
            config["bucket"], 
            config["prefix"], 
            config["batch_size"]
        )
        
        # Print final statistics
        stats = cache_manager.get_statistics()
        logger.info("=== Final Statistics ===")
        logger.info(json.dumps(stats, indent=2))
        
        logger.info("=== GCS Cache Refresh Completed Successfully ===")
        
    except KeyboardInterrupt:
        logger.warning("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.log_structured_error("MAIN_EXECUTION_ERROR", "main", e)
        logger.error(f"Cache refresh failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()