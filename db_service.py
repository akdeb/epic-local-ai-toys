import sqlite3
import json
import uuid
import time
import os
import glob
import random
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

def _argv_value(flag: str) -> Optional[str]:
    try:
        i = sys.argv.index(flag)
    except ValueError:
        return None
    if i + 1 >= len(sys.argv):
        return None
    v = sys.argv[i + 1].strip()
    return v if v else None

DEFAULT_DB_PATH = os.path.join(os.getcwd(), "elato.db")

DB_PATH = (
    _argv_value("--db-path")
    or _argv_value("--db_path")
    or os.environ.get("ELATO_DB_PATH")
    or DEFAULT_DB_PATH
)

@dataclass
class Personality:
    id: str
    name: str
    prompt: str
    short_description: str
    tags: List[str]
    is_visible: bool
    voice_id: str  # Added to map to tts voice (dave, jo, mara, santa)

@dataclass
class Conversation:
    id: str
    role: str  # 'user' or 'ai'
    transcript: str
    timestamp: float
    session_id: Optional[str] = None

@dataclass
class User:
    id: str
    name: str
    age: Optional[int]
    dob: Optional[str]
    hobbies: List[str]
    personality_type: Optional[str]
    likes: List[str]
    current_personality_id: Optional[str]
    user_type: str = "family"
    device_volume: int = 70

@dataclass
class Session:
    id: str  # session_id
    started_at: float
    ended_at: Optional[float]
    duration_sec: Optional[float]
    client_type: str  # 'computer' | 'device'
    user_id: Optional[str]
    personality_id: Optional[str]

class DBService:
    def __init__(self, db_path: str = DB_PATH):
        if not db_path:
            db_path = DEFAULT_DB_PATH
        self.db_path = db_path
        if self.db_path != ":memory:":
            try:
                Path(self.db_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
        self._maybe_reset_db()
        self._apply_migrations()
        self._seed_defaults()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _maybe_reset_db(self) -> None:
        if os.environ.get("ELATO_WIPE_DB", "0") != "1":
            return
        try:
            if os.path.exists(self.db_path):
                os.remove(self.db_path)
            wal = f"{self.db_path}-wal"
            shm = f"{self.db_path}-shm"
            if os.path.exists(wal):
                os.remove(wal)
            if os.path.exists(shm):
                os.remove(shm)
        except Exception:
            pass

    def _apply_migrations(self) -> None:
        migrations_dir = os.environ.get(
            "ELATO_MIGRATIONS_DIR",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "migrations"),
        )

        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute("PRAGMA journal_mode=WAL;")
        cursor.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations (version TEXT PRIMARY KEY, applied_at REAL NOT NULL)"
        )

        cursor.execute("SELECT version FROM schema_migrations")
        applied = {row["version"] for row in cursor.fetchall()}

        files = sorted(glob.glob(os.path.join(migrations_dir, "*.sql")))
        for path in files:
            version = os.path.basename(path)
            if version in applied:
                continue
            with open(path, "r", encoding="utf-8") as f:
                sql = f.read().strip()
            if sql:
                cursor.executescript(sql)
            cursor.execute(
                "INSERT INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                (version, time.time()),
            )

        conn.commit()
        conn.close()

    def get_active_user_id(self) -> Optional[str]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM app_state WHERE key = ?", ("active_user_id",))
        row = cursor.fetchone()
        conn.close()
        return row["value"] if row and row["value"] else None

    def set_active_user_id(self, user_id: Optional[str]) -> None:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO app_state (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            ("active_user_id", user_id)
        )
        conn.commit()
        conn.close()

    def get_app_mode(self) -> str:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM app_state WHERE key = ?", ("app_mode",))
        row = cursor.fetchone()
        conn.close()
        return (row["value"] if row and row["value"] else None) or "idle"

    def set_app_mode(self, mode: Optional[str]) -> str:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO app_state (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            ("app_mode", mode or "idle")
        )
        conn.commit()
        conn.close()
        return self.get_app_mode()

    def get_device_status(self) -> Dict[str, Any]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT key, value FROM app_state WHERE key IN (?, ?)", ("esp32_connected", "esp32_session_id"))
        rows = cursor.fetchall()
        conn.close()

        data = {row["key"]: row["value"] for row in rows}
        connected = (data.get("esp32_connected") or "0") == "1"
        return {
            "connected": connected,
            "session_id": data.get("esp32_session_id") if connected else None,
        }

    def set_device_status(self, connected: bool, session_id: Optional[str] = None) -> None:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO app_state (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            ("esp32_connected", "1" if connected else "0")
        )
        cursor.execute(
            "INSERT INTO app_state (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            ("esp32_session_id", session_id if connected else None)
        )
        conn.commit()
        conn.close()

    def start_session(self, session_id: str, client_type: str, user_id: Optional[str] = None, personality_id: Optional[str] = None) -> None:
        if user_id is None:
            user_id = self.get_active_user_id()
        conn = self._get_conn()
        cursor = conn.cursor()
        started_at = time.time()

        cursor.execute("PRAGMA table_info(sessions)")
        session_columns = [row["name"] for row in cursor.fetchall()]

        if "channel" in session_columns:
            cursor.execute(
                "INSERT OR IGNORE INTO sessions (id, started_at, ended_at, duration_sec, client_type, channel, user_id, personality_id) VALUES (?, ?, NULL, NULL, ?, ?, ?, ?)",
                (session_id, started_at, client_type, "chat", user_id, personality_id)
            )
        else:
            cursor.execute(
                "INSERT OR IGNORE INTO sessions (id, started_at, ended_at, duration_sec, client_type, user_id, personality_id) VALUES (?, ?, NULL, NULL, ?, ?, ?)",
                (session_id, started_at, client_type, user_id, personality_id)
            )

        if user_id is not None or personality_id is not None:
            cursor.execute(
                "UPDATE sessions SET user_id = COALESCE(user_id, ?), personality_id = COALESCE(personality_id, ?) WHERE id = ?",
                (user_id, personality_id, session_id)
            )
        conn.commit()
        conn.close()

    def end_session(self, session_id: str) -> None:
        conn = self._get_conn()
        cursor = conn.cursor()
        ended_at = time.time()
        cursor.execute("SELECT started_at, ended_at FROM sessions WHERE id = ?", (session_id,))
        row = cursor.fetchone()
        if row and row["started_at"] and not row["ended_at"]:
            duration = ended_at - row["started_at"]
            cursor.execute(
                "UPDATE sessions SET ended_at = ?, duration_sec = ? WHERE id = ?",
                (ended_at, duration, session_id)
            )
        conn.commit()
        conn.close()

    def get_sessions(self, limit: int = 50, offset: int = 0, user_id: Optional[str] = None) -> List[Session]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(sessions)")
        session_columns = [row["name"] for row in cursor.fetchall()]

        if user_id and "user_id" in session_columns:
            cursor.execute(
                "SELECT * FROM sessions WHERE user_id = ? ORDER BY started_at DESC LIMIT ? OFFSET ?",
                (user_id, limit, offset)
            )
        else:
            cursor.execute(
                "SELECT * FROM sessions ORDER BY started_at DESC LIMIT ? OFFSET ?",
                (limit, offset)
            )
        rows = cursor.fetchall()
        conn.close()

        items: List[Session] = []
        for row in rows:
            items.append(
                Session(
                    id=row["id"],
                    started_at=row["started_at"],
                    ended_at=row["ended_at"],
                    duration_sec=row["duration_sec"],
                    client_type=row["client_type"],
                    user_id=row["user_id"] if "user_id" in session_columns else None,
                    personality_id=row["personality_id"] if "personality_id" in session_columns else None,
                )
            )
        return items

    def _seed_defaults(self):
        defaults = [
            {
                "name": "Dave",
                "prompt": "You are Dave, a helpful and knowledgeable AI assistant. You speak clearly and concisely.",
                "short_description": "Standard British male voice, helpful and precise.",
                "tags": ["assistant", "british", "male", "standard"],
                "voice_id": "dave"
            },
            {
                "name": "Jo",
                "prompt": "You are Jo, a friendly and casual AI companion. You like to keep things light and conversational.",
                "short_description": "Friendly American female voice, casual tone.",
                "tags": ["companion", "american", "female", "casual"],
                "voice_id": "jo"
            },
            {
                "name": "Mara",
                "prompt": "You are Mara, a professional and articulate assistant. You are efficient and get straight to the point.",
                "short_description": "Professional female voice, articulate and efficient.",
                "tags": ["professional", "female", "assistant", "articulate"],
                "voice_id": "mara"
            },
            {
                "name": "Santa",
                "prompt": "Ho ho ho! You are Santa Claus. You are jolly, festive, and full of holiday cheer. You love talking about reindeer, elves, and Christmas spirit.",
                "short_description": "Jolly Santa Claus voice, festive and cheerful.",
                "tags": ["festive", "character", "male", "holiday"],
                "voice_id": "santa"
            }
        ]

        conn = self._get_conn()
        cursor = conn.cursor()
        
        for p in defaults:
            # Check if exists by voice_id to avoid duplicates on restart
            cursor.execute("SELECT id FROM personalities WHERE voice_id = ?", (p["voice_id"],))
            if not cursor.fetchone():
                p_id = str(uuid.uuid4())
                cursor.execute(
                    "INSERT INTO personalities (id, name, prompt, short_description, tags, is_visible, voice_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (p_id, p["name"], p["prompt"], p["short_description"], json.dumps(p["tags"]), True, p["voice_id"])
                )

        cursor.execute("SELECT COUNT(1) AS n FROM users")
        row = cursor.fetchone()
        has_users = bool(row and row["n"])
        if not has_users:
            cursor.execute("SELECT id FROM personalities WHERE voice_id = ?", ("dave",))
            p_row = cursor.fetchone()
            default_personality_id = p_row["id"] if p_row else None
            default_user_id = str(uuid.uuid4())

            hobbies_csv = os.environ.get("ELATO_DEFAULT_USER_HOBBIES", "reading,lego,science")
            default_hobbies = [h.strip() for h in hobbies_csv.split(",") if h.strip()]

            default_user_name = "Elato"
            cursor.execute(
                """INSERT INTO users (id, name, age, dob, hobbies, personality_type, likes, current_personality_id, user_type, device_volume)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (default_user_id, default_user_name, 11, None, json.dumps(default_hobbies), None, json.dumps([]), default_personality_id, "family", 70)
            )

        conn.commit()
        conn.close()

        if not self.get_active_user_id():
            conn = self._get_conn()
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM users ORDER BY rowid ASC LIMIT 1")
            u = cursor.fetchone()
            conn.close()
            if u and u["id"]:
                self.set_active_user_id(u["id"])

    # --- Personalities CRUD ---

    def create_personality(self, name: str, prompt: str, short_description: str, tags: List[str], voice_id: str, is_visible: bool = True) -> Personality:
        p_id = str(uuid.uuid4())
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO personalities (id, name, prompt, short_description, tags, is_visible, voice_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (p_id, name, prompt, short_description, json.dumps(tags), is_visible, voice_id)
        )
        conn.commit()
        conn.close()
        return Personality(p_id, name, prompt, short_description, tags, is_visible, voice_id)

    def get_personalities(self, include_hidden: bool = False) -> List[Personality]:
        conn = self._get_conn()
        cursor = conn.cursor()
        if include_hidden:
            cursor.execute("SELECT * FROM personalities")
        else:
            cursor.execute("SELECT * FROM personalities WHERE is_visible = 1")
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            Personality(
                id=row["id"],
                name=row["name"],
                prompt=row["prompt"],
                short_description=row["short_description"],
                tags=json.loads(row["tags"]),
                is_visible=bool(row["is_visible"]),
                voice_id=row["voice_id"]
            )
            for row in rows
        ]

    def get_personality(self, p_id: str) -> Optional[Personality]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM personalities WHERE id = ?", (p_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return Personality(
                id=row["id"],
                name=row["name"],
                prompt=row["prompt"],
                short_description=row["short_description"],
                tags=json.loads(row["tags"]),
                is_visible=bool(row["is_visible"]),
                voice_id=row["voice_id"]
            )
        return None
    
    def get_personality_by_voice(self, voice_id: str) -> Optional[Personality]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM personalities WHERE voice_id = ?", (voice_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return Personality(
                id=row["id"],
                name=row["name"],
                prompt=row["prompt"],
                short_description=row["short_description"],
                tags=json.loads(row["tags"]),
                is_visible=bool(row["is_visible"]),
                voice_id=row["voice_id"]
            )
        return None

    def update_personality(self, p_id: str, **kwargs) -> Optional[Personality]:
        current = self.get_personality(p_id)
        if not current:
            return None
            
        # Prepare updates
        fields = []
        values = []
        
        if "name" in kwargs:
            fields.append("name = ?")
            values.append(kwargs["name"])
        if "prompt" in kwargs:
            fields.append("prompt = ?")
            values.append(kwargs["prompt"])
        if "short_description" in kwargs:
            fields.append("short_description = ?")
            values.append(kwargs["short_description"])
        if "tags" in kwargs:
            fields.append("tags = ?")
            values.append(json.dumps(kwargs["tags"]))
        if "is_visible" in kwargs:
            fields.append("is_visible = ?")
            values.append(kwargs["is_visible"])
        if "voice_id" in kwargs:
            fields.append("voice_id = ?")
            values.append(kwargs["voice_id"])
            
        if not fields:
            return current
            
        values.append(p_id)
        query = f"UPDATE personalities SET {', '.join(fields)} WHERE id = ?"
        
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(query, tuple(values))
        conn.commit()
        conn.close()
        
        return self.get_personality(p_id)

    def delete_personality(self, p_id: str) -> bool:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM personalities WHERE id = ?", (p_id,))
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return success

    # --- Conversations CRUD ---

    def log_conversation(self, role: str, transcript: str, session_id: Optional[str] = None) -> Conversation:
        c_id = str(uuid.uuid4())
        timestamp = time.time()
        
        conn = self._get_conn()
        cursor = conn.cursor()

        cursor.execute(
            "INSERT INTO conversations (id, role, transcript, timestamp, session_id) VALUES (?, ?, ?, ?, ?)",
            (c_id, role, transcript, timestamp, session_id)
        )
        conn.commit()
        conn.close()
        
        return Conversation(c_id, role, transcript, timestamp, session_id)

    def get_conversations(self, limit: int = 50, offset: int = 0, session_id: Optional[str] = None) -> List[Conversation]:
        conn = self._get_conn()
        cursor = conn.cursor()

        if session_id:
            cursor.execute(
                "SELECT * FROM conversations WHERE session_id = ? ORDER BY timestamp ASC",
                (session_id,)
            )
        else:
            cursor.execute(
                "SELECT * FROM conversations ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                (limit, offset)
            )
        rows = cursor.fetchall()
        conn.close()
        
        return [
            Conversation(
                id=row["id"],
                role=row["role"],
                transcript=row["transcript"],
                timestamp=row["timestamp"],
                session_id=row["session_id"] if "session_id" in row.keys() else None
            )
            for row in rows
        ]

    def delete_conversation(self, c_id: str) -> bool:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM conversations WHERE id = ?", (c_id,))
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return success

    def clear_conversations(self):
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM conversations")
        conn.commit()
        conn.close()

    # --- User CRUD ---

    def create_user(self, name: str, age: Optional[int] = None, dob: Optional[str] = None, 
                   hobbies: List[str] = [], personality_type: Optional[str] = None, 
                   likes: List[str] = [], current_personality_id: Optional[str] = None,
                   user_type: str = "family", device_volume: int = 70) -> User:
        u_id = str(uuid.uuid4())
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO users (id, name, age, dob, hobbies, personality_type, likes, current_personality_id, user_type, device_volume) 
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (u_id, name, age, dob, json.dumps(hobbies), personality_type, json.dumps(likes), current_personality_id, user_type, device_volume)
        )
        conn.commit()
        conn.close()
        return User(u_id, name, age, dob, hobbies, personality_type, likes, current_personality_id, user_type, device_volume)

    def get_users(self) -> List[User]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users")
        rows = cursor.fetchall()
        conn.close()
        
        return [
            User(
                id=row["id"],
                name=row["name"],
                age=row["age"],
                dob=row["dob"],
                hobbies=json.loads(row["hobbies"]) if row["hobbies"] else [],
                personality_type=row["personality_type"],
                likes=json.loads(row["likes"]) if row["likes"] else [],
                current_personality_id=row["current_personality_id"],
                user_type=row["user_type"] if "user_type" in row.keys() else "family",
                device_volume=row["device_volume"] if "device_volume" in row.keys() else 70
            )
            for row in rows
        ]

    def get_user(self, u_id: str) -> Optional[User]:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE id = ?", (u_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return User(
                id=row["id"],
                name=row["name"],
                age=row["age"],
                dob=row["dob"],
                hobbies=json.loads(row["hobbies"]) if row["hobbies"] else [],
                personality_type=row["personality_type"],
                likes=json.loads(row["likes"]) if row["likes"] else [],
                current_personality_id=row["current_personality_id"],
                user_type=row["user_type"] if "user_type" in row.keys() else "family",
                device_volume=row["device_volume"] if "device_volume" in row.keys() else 70
            )
        return None

    def update_user(self, u_id: str, **kwargs) -> Optional[User]:
        current = self.get_user(u_id)
        if not current:
            return None
            
        fields = []
        values = []
        
        if "name" in kwargs:
            fields.append("name = ?")
            values.append(kwargs["name"])
        if "age" in kwargs:
            fields.append("age = ?")
            values.append(kwargs["age"])
        if "dob" in kwargs:
            fields.append("dob = ?")
            values.append(kwargs["dob"])
        if "hobbies" in kwargs:
            fields.append("hobbies = ?")
            values.append(json.dumps(kwargs["hobbies"]))
        if "personality_type" in kwargs:
            fields.append("personality_type = ?")
            values.append(kwargs["personality_type"])
        if "likes" in kwargs:
            fields.append("likes = ?")
            values.append(json.dumps(kwargs["likes"]))
        if "current_personality_id" in kwargs:
            fields.append("current_personality_id = ?")
            values.append(kwargs["current_personality_id"])
        if "user_type" in kwargs:
            fields.append("user_type = ?")
            values.append(kwargs["user_type"])
        if "device_volume" in kwargs:
            fields.append("device_volume = ?")
            values.append(kwargs["device_volume"])
            
        if not fields:
            return current
            
        values.append(u_id)
        query = f"UPDATE users SET {', '.join(fields)} WHERE id = ?"
        
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute(query, tuple(values))
        conn.commit()
        conn.close()
        
        return self.get_user(u_id)

    def delete_user(self, u_id: str) -> bool:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM users WHERE id = ?", (u_id,))
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return success

db_service = DBService()
