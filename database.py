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
    logger.error("CRITIQUE: SUPABASE_URL et SUPABASE_KEY sont manquantes.")
    supabase = None

AUTONOMY_LEVELS = ("ask", "draft", "auto")

# --- Compatibilité pour les anciens fichiers (comme proposals.py) ---
USE_PG = False

@contextmanager
def get_db():
    class DummyConn:
        def execute(self, query, params=()): pass
        def fetchone(self): return None
        def fetchall(self): return []
    yield DummyConn()
# -------------------------------------------------------------------

def init_db():
    if supabase:
        logger.info("Base de données initialisée avec Supabase (table: luce_users).")
    else:
        logger.error("Impossible d'initialiser Supabase : variables d'environnement manquantes.")

def create_user(user_id: str):
    if not supabase: return
    supabase.table("luce_users").upsert(
        {"id": user_id, "autonomy": "ask"}, 
        on_conflict="id"
    ).execute()

def delete_user(user_id: str):
    if not supabase: return
    supabase.table("luce_users").delete().eq("id", user_id).execute()

def get_autonomy(user_id: str) -> str:
    if not supabase: return "ask"
    res = supabase.table("luce_users").select("autonomy").eq("id", user_id).single().execute()
    if res.data and res.data.get("autonomy") in AUTONOMY_LEVELS:
        return res.data["autonomy"]
    return "ask"

def set_autonomy(user_id: str, level: str):
    if not supabase: return
    if level not in AUTONOMY_LEVELS:
        raise ValueError("invalid autonomy level")
    supabase.table("luce_users").update({"autonomy": level}).eq("id", user_id).execute()

def get_composio_session_id(user_id: str):
    if not supabase: return None
    res = supabase.table("luce_users").select("composio_session").eq("id", user_id).single().execute()
    return res.data.get("composio_session") if res.data else None

def save_composio_session_id(user_id: str, session_id: str):
    if not supabase: return
    supabase.table("luce_users").update({"composio_session": session_id}).eq("id", user_id).execute()

def save_message(user_id: str, role: str, content: str):
    if not supabase: return
    try:
        supabase.table("messages").insert({"user_id": user_id, "role": role, "content": content}).execute()
    except Exception:
        pass  # Ignore si la table n'existe pas encore

def get_messages(user_id: str, limit: int | None = None):
    if not supabase: return []
    try:
        if limit:
            res = supabase.table("messages").select("role, content").eq("user_id", user_id).order("id", desc=True).limit(limit).execute()
            if res.data:
                return [{"role": r["role"], "content": r["content"]} for r in reversed(res.data)]
            return []
        res = supabase.table("messages").select("role, content").eq("user_id", user_id).order("id", desc=False).execute()
        return [{"role": r["role"], "content": r["content"]} for r in res.data] if res.data else []
    except Exception:
        return []

def clear_messages(user_id: str):
    if not supabase: return
    try:
        supabase.table("messages").delete().eq("user_id", user_id).execute()
    except Exception:
        pass

def create_pending_action(user_id: str, tool: str, arguments: dict) -> str:
    if not supabase: return uuid.uuid4().hex
    action_id = uuid.uuid4().hex
    try:
        supabase.table("pending_actions").insert({
            "id": action_id, "user_id": user_id, "tool": tool,
            "arguments": json.dumps(arguments, default=str), "status": "pending"
        }).execute()
    except Exception:
        pass
    return action_id

def _action_row(row: dict):
    return {
        "id": row["id"], "tool": row["tool"],
        "arguments": json.loads(row["arguments"]) if isinstance(row["arguments"], str) else row["arguments"],
        "status": row["status"], "created_at": str(row.get("created_at", "")),
    }

def list_pending_actions(user_id: str):
    if not supabase: return []
    try:
        res = supabase.table("pending_actions").select("*").eq("user_id", user_id).eq("status", "pending").order("created_at", desc=False).execute()
        return [_action_row(r) for r in res.data] if res.data else []
    except Exception:
        return []

def get_pending_action(user_id: str, action_id: str):
    if not supabase: return None
    try:
        res = supabase.table("pending_actions").select("*").eq("id", action_id).eq("user_id", user_id).single().execute()
        return _action_row(res.data) if res.data else None
    except Exception:
        return None

def claim_pending_action(user_id: str, action_id: str, new_status: str) -> bool:
    if not supabase: return False
    try:
        res = supabase.table("pending_actions").update({"status": new_status}).eq("id", action_id).eq("user_id", user_id).eq("status", "pending").execute()
        return len(res.data) == 1
    except Exception:
        return False
