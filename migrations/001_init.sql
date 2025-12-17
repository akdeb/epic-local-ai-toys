PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS schema_migrations (
  version TEXT PRIMARY KEY,
  applied_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS app_state (
  key TEXT PRIMARY KEY,
  value TEXT
);

CREATE TABLE IF NOT EXISTS sessions (
  id TEXT PRIMARY KEY,
  started_at REAL NOT NULL,
  ended_at REAL,
  duration_sec REAL,
  client_type TEXT NOT NULL,
  user_id TEXT,
  personality_id TEXT
);

CREATE TABLE IF NOT EXISTS personalities (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  prompt TEXT NOT NULL,
  short_description TEXT,
  tags TEXT,
  is_visible BOOLEAN DEFAULT 1,
  voice_id TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS conversations (
  id TEXT PRIMARY KEY,
  role TEXT NOT NULL,
  transcript TEXT NOT NULL,
  timestamp REAL NOT NULL,
  session_id TEXT
);

CREATE TABLE IF NOT EXISTS users (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  age INTEGER,
  dob TEXT,
  hobbies TEXT,
  personality_type TEXT,
  likes TEXT,
  current_personality_id TEXT,
  user_type TEXT DEFAULT 'family',
  device_volume INTEGER DEFAULT 70,
  FOREIGN KEY (current_personality_id) REFERENCES personalities (id)
);
