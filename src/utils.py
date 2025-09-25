import sqlite3, json
from datetime import datetime

def ensure_schema_sqlite(db_path: str, schema_sql_path: str = "db/schema.sql"):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cur = conn.cursor()
    with open(schema_sql_path, "r") as f:
        cur.executescript(f.read())
    conn.commit()
    conn.close()

def log_detection(db_path: str, camera_id: str, cls: str, conf: float, bbox, image_path=None):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cur = conn.cursor()
    cur.execute("""
      INSERT INTO detections (timestamp, camera_id, class, confidence, bbox, image_path)
      VALUES (?, ?, ?, ?, ?, ?)
    """, (datetime.utcnow().isoformat(), camera_id, cls, float(conf), json.dumps(bbox), image_path))
    conn.commit()
    conn.close()
