import json
import logging
import os
import uuid
from supabase import create_client, Client

logger = logging.getLogger("luce.db")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL et SUPABASE_KEY sont manquantes dans les variables d'environnement.")

# Initialisation du client Supabase
# Astuce : Utilise la clé "service_role" (et non "anon") dans Vercel pour que 
# le backend ait tous les droits d'écriture sans être bloqué par la RLS de Supabase.
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

AUTONOMY_LEVELS = ("ask", "draft", "auto")


def init_db():
    """
    Supabase gère le schéma via l'éditeur SQL. 
    Cette fonction est conservée pour la compatibilité mais n'a plus besoin de créer les tables.
    """
    logger.info("Base de données initialisée avec Supabase.")


# ---- users -----------------------------------------------------------------

def create_user(user_id: str):
    # Upsert pour éviter les erreurs si l'utilisateur existe déjà
    supabase.table("users").upsert(
        {"id": user_id, "autonomy": "ask"}, 
        on_conflict="id"
    ).execute()


def delete_user(user_id: str):
    """Supprime l'utilisateur (la cascade en SQL supprimera aussi sessions, messages et actions)."""
    supabase.table("users").delete().eq("id", user_id).execute()


def get_autonomy(user_id: str) -> str:
    res = supabase.table("users").select("autonomy").eq("id", user_id).single().execute()
    if res.data and res.data.get("autonomy") in AUTONOMY_LEVELS:
        return res.data["autonomy"]
    return "ask"


def set_autonomy(user_id: str, level: str):
    if level not in AUTONOMY_LEVELS:
        raise ValueError("invalid autonomy level")
    supabase.table("users").update({"autonomy": level}).eq("id", user_id).execute()


# ---- composio session ------------------------------------------------------

def get_composio_session_id(user_id: str):
    res = supabase.table("conversations").select("composio_session_id").eq("user_id", user_id).single().execute()
    return res.data.get("composio_session_id") if res.data else None


def save_composio_session_id(user_id: str, session_id: str):
    supabase.table("conversations").upsert(
        {"user_id": user_id, "composio_session_id": session_id}, 
        on_conflict="user_id"
    ).execute()


# ---- messages --------------------------------------------------------------

def save_message(user_id: str, role: str, content: str):
    supabase.table("messages").insert({
        "user_id": user_id,
        "role": role,
        "content": content
    }).execute()


def get_messages(user_id: str, limit: int | None = None):
    if limit:
        # On récupère les derniers messages, puis on les inverse en Python pour les avoir du plus ancien au plus récent
        res = supabase.table("messages").select("role, content").eq("user_id", user_id).order("id", desc=True).limit(limit).execute()
        if res.data:
            return [{"role": r["role"], "content": r["content"]} for r in reversed(res.data)]
        return []
    
    res = supabase.table("messages").select("role, content").eq("user_id", user_id).order("id", desc=False).execute()
    return [{"role": r["role"], "content": r["content"]} for r in res.data] if res.data else []


def clear_messages(user_id: str):
    supabase.table("messages").delete().eq("user_id", user_id).execute()


# ---- pending actions (human confirmation) ----------------------------------

def create_pending_action(user_id: str, tool: str, arguments: dict) -> str:
    action_id = uuid.uuid4().hex
    supabase.table("pending_actions").insert({
        "id": action_id,
        "user_id": user_id,
        "tool": tool,
        "arguments": json.dumps(arguments, default=str),
        "status": "pending"
    }).execute()
    return action_id


def _action_row(row: dict):
    return {
        "id": row["id"],
        "tool": row["tool"],
        "arguments": json.loads(row["arguments"]) if isinstance(row["arguments"], str) else row["arguments"],
        "status": row["status"],
        "created_at": str(row.get("created_at", "")),
    }


def list_pending_actions(user_id: str):
    res = supabase.table("pending_actions").select("*").eq("user_id", user_id).eq("status", "pending").order("created_at", desc=False).execute()
    return [_action_row(r) for r in res.data] if res.data else []


def get_pending_action(user_id: str, action_id: str):
    res = supabase.table("pending_actions").select("*").eq("id", action_id).eq("user_id", user_id).single().execute()
    return _action_row(res.data) if res.data else None


def claim_pending_action(user_id: str, action_id: str, new_status: str) -> bool:
    """
    Met à jour le statut de 'pending' à new_status de manière atomique.
    Retourne True si une ligne a été modifiée, False sinon (évite les doubles clics).
    """
    res = supabase.table("pending_actions").update({"status": new_status}).eq("id", action_id).eq("user_id", user_id).eq("status", "pending").execute()
    return len(res.data) == 1
