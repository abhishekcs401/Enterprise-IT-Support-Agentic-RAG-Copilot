import json
import sqlite3
from datetime import datetime, timezone
from app.core.config import get_settings

settings = get_settings()


def get_connection():
    db_path = settings.audit_db_path
    from pathlib import Path
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)


def init_db() -> None:
    with get_connection() as con:
        con.execute("""
            CREATE TABLE IF NOT EXISTS query_audit (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                question TEXT NOT NULL,
                source_used TEXT NOT NULL,
                trace_json TEXT NOT NULL
            )
        """)

        con.execute("""
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

        con.execute("""
            CREATE INDEX IF NOT EXISTS idx_chat_session
            ON chat_messages(session_id, id)
        """)



def write_audit(question: str, source_used: str, trace: list[str]) -> None:
    with get_connection() as con:
        con.execute(
            """INSERT INTO query_audit
            (created_at, question, source_used, trace_json)
            VALUES (?, ?, ?, ?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                question,
                source_used,
                json.dumps(trace),
            ),
        )

def save_chat_message(session_id: str, role: str, content: str) -> None:
    with get_connection() as con:
        con.execute(
            """INSERT INTO chat_messages
            (session_id, role, content, created_at)
            VALUES (?, ?, ?, ?)""",
            (
                session_id,
                role,
                content,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
def get_chat_history(session_id: str, limit: int = 20) -> list[dict]:
    with get_connection() as con:
        rows = con.execute(
            """SELECT role, content
            FROM chat_messages
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?""",
            (session_id, limit),
        ).fetchall()

    return [
        {"role": role, "content": content}
        for role, content in reversed(rows)
    ]
