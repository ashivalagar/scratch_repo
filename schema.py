from utilities.caching.sqlite_caching import SQLiteBaseCache

db_path = r"\\fsnvemaffs\nve_maf\cache\cache.db"
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
