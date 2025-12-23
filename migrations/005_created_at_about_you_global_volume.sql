PRAGMA foreign_keys=OFF;

-- Add created_at for ordering
ALTER TABLE voices ADD COLUMN created_at REAL;
UPDATE voices SET created_at = COALESCE(created_at, strftime('%s','now'));

ALTER TABLE personalities ADD COLUMN created_at REAL;
UPDATE personalities SET created_at = COALESCE(created_at, strftime('%s','now'));

-- Replace hobbies tags with a single about_you string (keep hobbies column for backward compatibility)
ALTER TABLE users ADD COLUMN about_you TEXT;
UPDATE users SET about_you = COALESCE(about_you, '');

-- Global (per-laptop) volume lives in app_state
INSERT INTO app_state (key, value)
VALUES ('laptop_volume', '70')
ON CONFLICT(key) DO NOTHING;

PRAGMA foreign_keys=ON;
