import json
import logging
import os
import uuid
from contextlib import contextmanager

logger = logging.getLogger("luce.db")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if SUPABASE_URL and SUPABASE_KEY:
    from supabase import create_client, Client
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    logger.error("CRITIQUE: SUPABASE_URL et SUPABASE_KEY sont manquantes dans les variables d'environnement Vercel.")
    supabase = None

AUTONOMY_LEVELS = ("ask", "draft", "auto")

# --- Compatibilité pour les anciens fichiers (comme proposals.py) ---
USE_PG = False

@contextmanager
def get_db():
    """Faux contexte de base de données pour éviter le crash de proposals.py à l'import."""
    class DummyConn:
        def execute(self, query, params=()): pass
        def fetchone(self): return None
        def fetchall(self): return []
    yield DummyConn()
# -------------------------------------------------------------------

def init_db():
    if supabase:
        logger.info("Base de données initialisée avec Supabase.")
    else:
        logger.error("Impossible d'initialiser Supabase : variables d'environnement manquantes.")

def create_user(user_id: str):
    if not supabase: raise RuntimeError("Supabase non configuré")
    supabase.table("users").upsert({"id": user_id, "autonomy": "ask"}, on_conflict="id").execute()

def delete_user(user_id: str):
    if not supabase: raise RuntimeError("Supabase non configuré")
    supabase.table("users").delete().eq("id", user_id).execute()

def get_autonomy(user_id: str) -> str:
    if not supabase: raise RuntimeError("Supabase non configuré")
    res = supabase.table("users").select("autonomy").eq("id", user_id).single().execute()
    if res.data and res.data.get("autonomy") in AUTONOMY_LEVELS:
        return res.data["autonomy"]
    return "ask"

def set_autonomy(user_id: str, level: str):
    if not supabase: raise RuntimeError("Supabase non configuré")
    if level not in AUTONOMY_LEVELS:
        raise ValueError("invalid autonomy level")
    supabase.table("users").update({"autonomy": level}).eq("id", user_id).execute()

def get_composio_session_id(user_id: str):
    if not supabase: raise RuntimeError("Supabase non configuré")
    res = supabase.table("conversations").select("composio_session_id").eq("user_id", user_id).single().execute()
    return res.data.get("composio_session_id") if res.data else None

def save_composio_session_id(user_id: str, session_id: str):
    if not supabase: raise RuntimeError("Supabase non configuré")
    supabase.table("conversations").upsert(
        {"user_id": user_id, "composio_session_id": session_id}, 
        on_conflict="user_id"
    ).execute()

def save_message(user_id: str, role: str, content: str):
    if not supabase: raise RuntimeError("Supabase non configuré")
    supabase.table("messages").insert({"user_id": user_id, "role": role, "content": content}).execute()

def get_messages(user_id: str, limit: int | None = None):
    if not supabase: raise RuntimeError("Supabase non configuré")
    if limit:
        res = supabase.table("messages").select("role, content").eq("user_id", user_id).order("id", desc=True).limit(limit).execute()
        if res.data:
            return [{"role": r["role"], "content": r["content"]} for r in reversed(res.data)]
        return []
    res = supabase.table("messages").select("role, content").eq("user_id", user_id).order("id", desc=False).execute()
    return [{"role": r["role"], "content": r["content"]} for r in res.data] if res.data else []

def clear_messages(user_id: str):
    if not supabase: raise RuntimeError("Supabase non configuré")
    supabase.table("messages").delete().eq("user_id", user_id).execute()

def create_pending_action(user_id: str, tool: str, arguments: dict) -> str:
    if not supabase: raise RuntimeError("Supabase non configuré")
    action_id = uuid.uuid4().hex
    supabase.table("pending_actions").insert({
        "id": action_id, "user_id": user_id, "tool": tool,
        "arguments": json.dumps(arguments, default=str), "status": "pending"
    }).execute()
    return action_id

def _action_row(row: dict):
    return {
        "id": row["id"], "tool": row["tool"],
        "arguments": json.loads(row["arguments"]) if isinstance(row["arguments"], str) else row["arguments"],
        "status": row["status"], "created_at": str(row.get("created_at", "")),
    }

def list_pending_actions(user_id: str):
    if not supabase: raise RuntimeError("Supabase non configuré")
    res = supabase.table("pending_actions").select("*").eq("user_id", user_id).eq("status", "pending").order("created_at", desc=False).execute()
    return [_action_row(r) for r in res.data] if res.data else []

def get_pending_action(user_id: str, action_id: str):
    if not supabase: raise RuntimeError("Supabase non configuré")
    res = supabase.table("pending_actions").select("*").eq("id", action_id).eq("user_id", user_id).single().execute()
    return _action_row(res.data) if res.data else None

def claim_pending_action(user_id: str, action_id: str, new_status: str) -> bool:
    if not supabase: raise RuntimeError("Supabase non configuré")
    res = supabase.table("pending_actions").update({"status": new_status}).eq("id", action_id).eq("user_id", user_id).eq("status", "pending").execute()
    return len(res.data) == 1
