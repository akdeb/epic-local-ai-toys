PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS voices (
  voice_id TEXT PRIMARY KEY,
  gender TEXT,
  voice_name TEXT NOT NULL,
  voice_description TEXT,
  voice_src TEXT,
  is_global BOOLEAN DEFAULT 0
);
