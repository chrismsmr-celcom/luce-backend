import logging
import os
import time
import uuid
from collections import defaultdict, deque
from datetime import timedelta
from threading import Lock

from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, request, session
from flask_cors import CORS

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

from agent import process_message, run_confirmed_action  # noqa: E402
import auth  # noqa: E402
from composio_service import authorize_toolkit, list_connected_accounts  # noqa: E402
from database import (  # noqa: E402
    AUTONOMY_LEVELS,
    claim_pending_action,
    clear_messages,
    create_user,
    delete_user,
    get_autonomy,
    get_messages,
    get_pending_action,
    init_db,
    list_pending_actions,
    set_autonomy,
)
from toolkits import TOOLKIT_SET, TOOLKITS  # noqa: E402

PRODUCTION = os.getenv("LUCE_ENV", "development").lower() == "production"

def _parse_origins(raw: str) -> list[str]:
    """Tolère les erreurs de saisie courantes dans la variable Vercel : guillemets, slash final,
    chemin (https://site.app/connexions), séparateurs , ; ou retours à la ligne."""
    from urllib.parse import urlparse

    origins: list[str] = []
    for item in raw.replace(";", ",").replace("\n", ",").split(","):
        item = item.strip().strip("\"'").strip()
        if not item:
            continue
        if "://" in item:
            parts = urlparse(item)
            item = f"{parts.scheme}://{parts.netloc}"
        origins.append(item.rstrip("/"))
    return origins


# FRONTEND_ORIGIN (singulier) accepté aussi : faute de frappe très fréquente.
ALLOWED_ORIGINS = _parse_origins(os.getenv("FRONTEND_ORIGINS") or os.getenv("FRONTEND_ORIGIN") or "")
FRONTEND_URL = os.getenv("FRONTEND_URL", "/connexions")
PUBLIC_BASE_URL = os.getenv("PUBLIC_BASE_URL", "").rstrip("/")

MAX_MESSAGE_CHARS = int(os.getenv("LUCE_MAX_MESSAGE_CHARS", "4000"))
CHAT_RATE_LIMIT = int(os.getenv("LUCE_CHAT_RATE_LIMIT", "20"))
CHAT_RATE_WINDOW = int(os.getenv("LUCE_CHAT_RATE_WINDOW", "60"))

app = Flask(__name__)

app.secret_key = os.getenv("FLASK_SECRET_KEY")
if not app.secret_key:
    raise RuntimeError("FLASK_SECRET_KEY is missing")

app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE=os.getenv("SESSION_COOKIE_SAMESITE", "None" if ALLOWED_ORIGINS else "Lax"),
    SESSION_COOKIE_SECURE=PRODUCTION,
    PERMANENT_SESSION_LIFETIME=timedelta(days=30),
    MAX_CONTENT_LENGTH=64 * 1024,
)

if ALLOWED_ORIGINS:
    CORS(
        app,
        resources={r"/api/*": {"origins": ALLOWED_ORIGINS}},
        supports_credentials=True,
        allow_headers=["Content-Type", "X-Luce-Client", "Authorization"],
    )

init_db()

@app.before_request
def csrf_guard():
    if request.method in ("GET", "HEAD", "OPTIONS") or not request.path.startswith("/api/"):
        return None
    if request.headers.get("X-Luce-Client") != "web":
        return jsonify({"error": "Missing client header"}), 403
    origin = request.headers.get("Origin")
    if origin:
        origin = origin.rstrip("/")
        same_origin = origin == request.host_url.rstrip("/")
        if not same_origin and origin not in ALLOWED_ORIGINS:
            return jsonify({"error": "Origin not allowed"}), 403
    return None

@app.after_request
def security_headers(resp):
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("Referrer-Policy", "no-referrer")
    resp.headers.setdefault("Cache-Control", "no-store")
    return resp

_hits: dict[str, deque] = defaultdict(deque)
_hits_lock = Lock()

def rate_limited(key: str, limit: int | None = None, window: int | None = None) -> bool:
    limit = CHAT_RATE_LIMIT if limit is None else limit
    window = CHAT_RATE_WINDOW if window is None else window
    now = time.monotonic()
    with _hits_lock:
        q = _hits[key]
        while q and now - q[0] > window:
            q.popleft()
        if len(q) >= limit:
            return True
        q.append(now)
        return False

_known_users: set[str] = set()

def get_user_id() -> str:
    if auth.ENABLED:
        user_id = auth.user_from_authorization(request.headers.get("Authorization"))
        if user_id not in _known_users:
            create_user(user_id)
            _known_users.add(user_id)
        return user_id

    if "user_id" not in session:
        user_id = "luce_" + uuid.uuid4().hex
        session.permanent = True
        session["user_id"] = user_id
        create_user(user_id)
    return session["user_id"]

@app.errorhandler(auth.AuthError)
def unauthorized(exc):
    return jsonify({"error": "Authentification requise", "reason": str(exc)}), 401

def callback_url() -> str:
    base = PUBLIC_BASE_URL or request.host_url.rstrip("/")
    return base + "/api/composio/callback"

def server_error(message: str, exc: Exception):
    app.logger.exception(message)
    return jsonify({"error": message}), 500

import data  # noqa: E402
data.register(app, get_user_id, server_error)

# AJOUT ESSENTIEL : Enregistrement des routes mail (détail et pièces jointes)
import mail  # noqa: E402
mail.register(app, get_user_id, server_error)

import proposals  # noqa: E402
proposals.register(app, get_user_id, server_error, rate_limited)

@app.get("/health")
@app.get("/api/health")
def health():
    # Diagnostic : montre ce que le serveur lit réellement (aucun secret). Si "cors_origins" est vide,
    # FRONTEND_ORIGINS est absente / mal saisie dans le projet Vercel du backend.
    return jsonify({"status": "ok", "service": "luce", "cors_origins": ALLOWED_ORIGINS, "auth": auth.ENABLED})

@app.get("/api/me")
def me():
    user_id = get_user_id()
    return jsonify(
        {
            "toolkits": TOOLKITS,
            "autonomy": get_autonomy(user_id),
            "auth": "supabase" if auth.ENABLED else "anonymous",
            "pending_actions": list_pending_actions(user_id),
        }
    )

@app.post("/api/connect/<toolkit>")
def connect_toolkit(toolkit):
    if toolkit not in TOOLKIT_SET:
        return jsonify({"error": "Unsupported toolkit"}), 400
    user_id = get_user_id()
    try:
        return jsonify(authorize_toolkit(user_id=user_id, toolkit=toolkit, callback_url=callback_url()))
    except Exception as exc:
        return server_error("Impossible de démarrer la connexion", exc)

@app.get("/api/composio/callback")
def composio_callback():
    return redirect(FRONTEND_URL)

@app.get("/api/connections")
def connections():
    user_id = get_user_id()
    try:
        result = {name: False for name in TOOLKITS}
        for account in list_connected_accounts(user_id):
            slug = str(getattr(getattr(account, "toolkit", None), "slug", "")).lower()
            if slug in result:
                result[slug] = True
        return jsonify(result)
    except Exception as exc:
        return server_error("Impossible de lire les connexions", exc)

@app.get("/api/history")
def history():
    return jsonify({"messages": get_messages(get_user_id())})

@app.post("/api/history/clear")
def history_clear():
    clear_messages(get_user_id())
    return jsonify({"success": True})

@app.post("/api/settings")
def settings():
    user_id = get_user_id()
    data = request.get_json(silent=True) or {}
    level = data.get("autonomy")
    if level not in AUTONOMY_LEVELS:
        return jsonify({"error": "Invalid autonomy level"}), 400
    set_autonomy(user_id, level)
    return jsonify({"autonomy": level})

@app.post("/api/chat")
def chat():
    user_id = get_user_id()

    if rate_limited(f"chat:{user_id}") or rate_limited(f"chat-ip:{request.remote_addr}", limit=CHAT_RATE_LIMIT * 3):
        return jsonify({"error": "Trop de requêtes, réessaie dans un instant."}), 429

    data = request.get_json(silent=True) or {}
    message = data.get("message")
    if not isinstance(message, str) or not message.strip():
        return jsonify({"error": "Message is required"}), 400
    message = message.strip()
    if len(message) > MAX_MESSAGE_CHARS:
        return jsonify({"error": f"Message trop long (max {MAX_MESSAGE_CHARS} caractères)"}), 413

    try:
        return jsonify(process_message(user_id=user_id, message=message))
    except Exception as exc:
        return server_error("Luce n'a pas pu traiter la demande", exc)

@app.get("/api/actions")
def actions():
    return jsonify({"pending_actions": list_pending_actions(get_user_id())})

@app.post("/api/actions/<action_id>/confirm")
def confirm_action(action_id):
    user_id = get_user_id()
    action = get_pending_action(user_id, action_id)
    if action is None:
        return jsonify({"error": "Action not found"}), 404
    if not claim_pending_action(user_id, action_id, "confirmed"):
        return jsonify({"error": "Action already handled"}), 409
    try:
        result = run_confirmed_action(user_id, action["tool"], action["arguments"])
    except Exception as exc:
        return server_error("L'action a échoué", exc)
    return jsonify(
        {
            "success": bool(result.get("success")),
            "pending_approval": bool(result.get("pending_approval")),
            "blocked": bool(result.get("blocked")),
            "error": result.get("error"),
        }
    )

@app.post("/api/actions/<action_id>/reject")
def reject_action(action_id):
    user_id = get_user_id()
    if get_pending_action(user_id, action_id) is None:
        return jsonify({"error": "Action not found"}), 404
    if not claim_pending_action(user_id, action_id, "rejected"):
        return jsonify({"error": "Action already handled"}), 409
    return jsonify({"success": True})

@app.post("/api/logout")
def logout():
    session.clear()
    return jsonify({"success": True})

@app.post("/api/account/delete")
def account_delete():
    user_id = get_user_id()
    delete_user(user_id)
    _known_users.discard(user_id)
    session.clear()
    return jsonify({"success": True})

@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(413)
def too_large(_):
    return jsonify({"error": "Request too large"}), 413

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "10000")), debug=False)

def register_snapshots(app):
    """Route Flask/FastAPI selon ton app.py (ici version Flask)."""

    from auth import require_user  # ton décorateur/middleware 401 existant

    @app.route("/api/data/snapshots")
    @require_user
    def api_tool_snapshots():
        from flask import jsonify
        try:
            return jsonify(build_snapshots(g.user_id)), 200
        except Exception:
            logger.exception("GET /api/data/snapshots failed")
            return jsonify({"error": "Failed to build snapshots"}), 500
