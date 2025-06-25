import sqlite3
from typing import List, Dict, Any, Tuple

class SQLiteBaseCache:
    def __init__(self, db_path: str, table_name: str, schema: Dict[str, str]):
        self.db_path = db_path
        self.table_name = table_name
        # Automatically add a default primary key
        self.schema = {"id": "INTEGER PRIMARY KEY AUTOINCREMENT", **schema}
        self._initialize_db()

    def _initialize_db(self):
        schema_str = ", ".join(f"{col} {dtype}" for col, dtype in self.schema.items())
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(f"CREATE TABLE IF NOT EXISTS {self.table_name} ({schema_str})")

    def insert_many(self, rows: List[Dict[str, Any]]):
        # Exclude 'id' from insert columns
        insert_columns = [col for col in self.schema if col != "id"]
        columns_str = ", ".join(insert_columns)
        placeholders = ", ".join(f":{col}" for col in insert_columns)

        # Prepare rows with only the insertable columns
        filtered_rows = [
            {col: row[col] for col in insert_columns if col in row}
            for row in rows
        ]

        with sqlite3.connect(self.db_path) as conn:
            conn.executemany(
                f"INSERT OR REPLACE INTO {self.table_name} ({columns_str}) VALUES ({placeholders})",
                filtered_rows
            )

    def query(self, sql: str, params: Tuple = ()) -> List[Tuple]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(sql, params)
            return cursor.fetchall()

    def clear(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(f"DELETE FROM {self.table_name}")
