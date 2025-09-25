CREATE TABLE IF NOT EXISTS detections (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp TEXT,
  camera_id TEXT,
  class TEXT,
  confidence REAL,
  bbox TEXT,       -- "[x1,y1,x2,y2]"
  image_path TEXT
);

CREATE INDEX IF NOT EXISTS idx_det_time ON detections (timestamp);
CREATE INDEX IF NOT EXISTS idx_det_class ON detections (class);
