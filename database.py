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
    class DummyResult:
        def fetchall(self): return []
        def fetchone(self): return None
    class DummyConn:
        def execute(self, query, params=()): return DummyResult()
        def commit(self): pass
        def rollback(self): pass
        def close(self): pass
    yield DummyConn()
# -------------------------------------------------------------------

def init_db():
    if supabase:
        logger.info("Base de données initialisée avec Supabase.")
    else:
        logger.error("Impossible d'initialiser Supabase : variables d'environnement manquantes.")

def create_user(user_id: str):
    if not supabase: return
    supabase.table("luce_users").upsert({"id": user_id, "autonomy": "ask"}, on_conflict="id").execute()

def delete_user(user_id: str):
    if not supabase: return
    supabase.table("luce_users").delete().eq("id", user_id).execute()

def get_autonomy(user_id: str) -> str:
    if not supabase:
        return "ask"
    try:
        res = (
            supabase.table("luce_users")
            .select("autonomy")
            .eq("id", user_id)
            .limit(1)
            .execute()
        )
        if res.data and res.data[0].get("autonomy") in AUTONOMY_LEVELS:
            return res.data[0]["autonomy"]
    except Exception as exc:
        logger.error("get_autonomy failed (user=%s): %s", user_id, exc)
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
        supabase.table("messages").insert(
            {"user_id": user_id, "role": role, "content": content}
        ).execute()
    except Exception as exc:
        # L'historique est un bonus, pas une dépendance critique :
        # on logge (visible dans Vercel) mais on ne casse pas la requête.
        logger.error("save_message failed (user=%s role=%s): %s",
                     user_id, role, exc)

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
    if not supabase: 
        logger.error("Supabase client is None. Cannot create pending action.")
        raise RuntimeError("Supabase non configuré")
    
    action_id = uuid.uuid4().hex
    try:
        # L'insertion doit réussir, sinon on lève l'exception pour voir l'erreur dans les logs Vercel
        res = supabase.table("pending_actions").insert({
            "id": action_id, 
            "user_id": user_id, 
            "tool": tool,
            "arguments": json.dumps(arguments, default=str), 
            "status": "pending"
        }).execute()
        logger.info("✅ Pending action %s créée avec succès pour l'utilisateur %s", action_id, user_id)
        return action_id
    except Exception as e:
        logger.error("❌ ÉCHEC CRITIQUE lors de la création de l'action en attente: %s", e)
        raise  # On ne cache plus l'erreur !
def save_memory(user_id: str, fact: str, category: str = "other"):
    """Enregistre un fait durable sur l'utilisateur (sans doublon exact)."""
    if not supabase:
        raise RuntimeError("Supabase non configuré")

    fact = (fact or "").strip()
    if not fact:
        return

    try:
        existing = (
            supabase.table("user_memories")
            .select("id")
            .eq("user_id", user_id)
            .eq("fact", fact)
            .limit(1)
            .execute()
        )
        if existing.data:
            return

        supabase.table("user_memories").insert(
            {"user_id": user_id, "fact": fact, "category": category}
        ).execute()
    except Exception as exc:
        logger.error("save_memory failed (user=%s): %s", user_id, exc)
        raise


def get_memories(user_id: str, limit: int = 40):
    """Renvoie les souvenirs les plus récents, du plus ancien au plus récent."""
    if not supabase:
        return []
    try:
        res = (
            supabase.table("user_memories")
            .select("fact, category")
            .eq("user_id", user_id)
            .order("id", desc=True)
            .limit(limit)
            .execute()
        )
        return list(reversed(res.data)) if res.data else []
    except Exception as exc:
        logger.error("get_memories failed (user=%s): %s", user_id, exc)
        return []


def clear_memories(user_id: str):
    """Efface toute la mémoire d'un utilisateur."""
    if not supabase:
        return
    try:
        supabase.table("user_memories").delete().eq("user_id", user_id).execute()
    except Exception as exc:
        logger.error("clear_memories failed (user=%s): %s", user_id, exc)
        
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
