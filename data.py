"""Lecture des vraies données (Gmail, Agenda, Drive) via Composio, pour les écrans Luce.

Branchement dans app.py (3 lignes, après la définition de get_user_id / server_error) :

    import data  # noqa: E402
    data.register(app, get_user_id, server_error)

Routes ajoutées (toutes authentifiées comme le reste de /api) :
    GET /api/data/inbox            -> derniers emails de la boîte de réception
    GET /api/data/agenda?tz=<min>  -> événements du jour (tz = Date.getTimezoneOffset() du navigateur)
    GET /api/data/files            -> fichiers récents de Google Drive

Ajoute ?raw=1 à n'importe laquelle pour voir la réponse brute de Composio
(utile si un champ n'est pas reconnu : colle-moi ce JSON).
"""
import json
import logging
from datetime import datetime, timedelta, timezone

from flask import Response, jsonify, request

from composio_service import execute_tool

logger = logging.getLogger("luce.data")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unwrap(result):
    """Normalise la réponse de session.execute -> (ok, data, error)."""
    if isinstance(result, dict):
        get = result.get
    else:
        def get(key, default=None):
            return getattr(result, key, default)

    data = get("data")
    if hasattr(data, "model_dump"):
        data = data.model_dump()
    error = get("error")
    ok = get("successful", get("success", error in (None, "")))
    return bool(ok), data, error


def _find_list(node, keys=("messages", "emails", "items", "files", "events", "results", "data")):
    """Cherche récursivement la première liste de dicts, en priorité sous `keys`."""
    if isinstance(node, list):
        return node if (not node or isinstance(node[0], dict)) else []
    if isinstance(node, dict):
        for k in keys:
            if k in node:
                found = _find_list(node[k], keys)
                if found or isinstance(node[k], list):
                    return found
        for v in node.values():
            if isinstance(v, (dict, list)):
                found = _find_list(v, keys)
                if found:
                    return found
    return []


def _run(user_id: str, slugs, attempts):
    """Essaie chaque slug avec des jeux d'arguments de plus en plus simples.

    Renvoie (data, raw_result). Lève la dernière exception si tout échoue.
    """
    last_exc = None
    for slug in slugs:
        for args in attempts:
            try:
                result = execute_tool(user_id=user_id, tool_slug=slug, arguments=args)
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning("%s %s a échoué: %s", slug, list(args), exc)
                continue
            ok, data, error = _unwrap(result)
            if ok:
                return data, result
            last_exc = RuntimeError(str(error) or f"{slug} a échoué")
            logger.warning("%s %s -> %s", slug, list(args), error)
    raise last_exc or RuntimeError("Aucun outil disponible")


def _raw_response(result):
    return Response(json.dumps(result, default=str, ensure_ascii=False, indent=2), mimetype="application/json")


def _text(value, limit=None):
    s = "" if value is None else str(value).strip()
    return s[:limit] if limit else s


# ---------------------------------------------------------------------------
# Mappers
# ---------------------------------------------------------------------------

def _map_email(m: dict) -> dict:
    labels = m.get("labelIds") or m.get("label_ids") or []
    preview = m.get("preview")
    if isinstance(preview, dict):
        preview = preview.get("body") or preview.get("subject")
    body = m.get("messageText") or m.get("message_text") or m.get("body") or ""
    snippet = preview or m.get("snippet") or body
    return {
        "id": _text(m.get("messageId") or m.get("message_id") or m.get("id")),
        "source": "gmail",
        "from": _text(m.get("sender") or m.get("from") or "Inconnu"),
        "subject": _text(m.get("subject") or (preview if isinstance(m.get("preview"), dict) else "") or "(sans objet)"),
        "preview": _text(snippet, 160),
        "body": _text(body, 6000),
        "date": _text(m.get("messageTimestamp") or m.get("message_timestamp") or m.get("date")),
        "unread": "UNREAD" in labels,
        "priority": "IMPORTANT" in labels or "STARRED" in labels,
    }


def _map_event(e: dict) -> dict:
    start = e.get("start") or {}
    end = e.get("end") or {}
    names = []
    for a in e.get("attendees") or []:
        if a.get("self"):
            continue
        names.append(a.get("displayName") or (a.get("email") or "").split("@")[0])
    return {
        "id": _text(e.get("id")),
        "title": _text(e.get("summary") or "(sans titre)"),
        "start": _text(start.get("dateTime") or start.get("date")),
        "end": _text(end.get("dateTime") or end.get("date")),
        "allDay": "date" in start and "dateTime" not in start,
        "who": ", ".join(n for n in names if n)[:80],
    }


def _file_type(mime: str) -> str:
    mime = (mime or "").lower()
    if "folder" in mime:
        return "folder"
    if "pdf" in mime:
        return "pdf"
    if "spreadsheet" in mime or "excel" in mime or "csv" in mime:
        return "sheet"
    if "presentation" in mime or "powerpoint" in mime:
        return "slides"
    if mime.startswith("image/"):
        return "image"
    return "doc"


def _map_file(f: dict) -> dict:
    owners = f.get("owners") or []
    owner = ""
    if owners and isinstance(owners[0], dict):
        owner = owners[0].get("displayName") or owners[0].get("emailAddress") or ""
    mime = f.get("mimeType") or f.get("mime_type") or ""
    size = f.get("size")
    return {
        "id": _text(f.get("id")),
        "name": _text(f.get("name") or f.get("title") or "(sans nom)"),
        "type": _file_type(mime),
        "owner": _text(owner),
        "modified": _text(f.get("modifiedTime") or f.get("modified_time")),
        "size": int(size) if str(size or "").isdigit() else None,
        "link": _text(f.get("webViewLink") or f.get("web_view_link")),
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

def register(app, get_user_id, server_error):
    raw = lambda: request.args.get("raw") == "1"  # noqa: E731

    @app.get("/api/data/inbox")
    def data_inbox():
        user_id = get_user_id()
        try:
            data, result = _run(
                user_id,
                ["GMAIL_FETCH_EMAILS"],
                [
                    {"max_results": 25, "label_ids": ["INBOX"], "verbose": True},
                    {"max_results": 25, "label_ids": ["INBOX"]},
                    {"max_results": 25},
                ],
            )
            if raw():
                return _raw_response(result if not isinstance(result, dict) else result)
            items = [_map_email(m) for m in _find_list(data, ("messages", "emails", "items"))]
            return jsonify({"items": [i for i in items if i["id"]]})
        except Exception as exc:  # noqa: BLE001
            return server_error("Impossible de lire Gmail (est-il connecté ?)", exc)

    @app.get("/api/data/agenda")
    def data_agenda():
        user_id = get_user_id()
        try:
            try:
                offset = int(request.args.get("tz", "0"))  # minutes, comme JS getTimezoneOffset()
            except ValueError:
                offset = 0
            local_now = datetime.now(timezone.utc) - timedelta(minutes=offset)
            start_local = local_now.replace(hour=0, minute=0, second=0, microsecond=0)
            start_utc = start_local + timedelta(minutes=offset)
            end_utc = start_utc + timedelta(days=1)
            fmt = "%Y-%m-%dT%H:%M:%SZ"
            args = {
                "calendarId": "primary",
                "timeMin": start_utc.strftime(fmt),
                "timeMax": end_utc.strftime(fmt),
                "singleEvents": True,
                "orderBy": "startTime",
                "maxResults": 20,
            }
            data, result = _run(
                user_id,
                ["GOOGLECALENDAR_EVENTS_LIST"],
                [args, {k: v for k, v in args.items() if k not in ("orderBy", "maxResults")}],
            )
            if raw():
                return _raw_response(result)
            items = [_map_event(e) for e in _find_list(data, ("items", "events"))]
            return jsonify({"items": items})
        except Exception as exc:  # noqa: BLE001
            return server_error("Impossible de lire l'agenda (Google Calendar est-il connecté ?)", exc)

    @app.get("/api/data/files")
    def data_files():
        user_id = get_user_id()
        try:
            data, result = _run(
                user_id,
                ["GOOGLEDRIVE_LIST_FILES", "GOOGLEDRIVE_FIND_FILE"],
                [{"pageSize": 40, "orderBy": "modifiedTime desc"}, {"page_size": 40}, {}],
            )
            if raw():
                return _raw_response(result)
            items = [_map_file(f) for f in _find_list(data, ("files", "items"))]
            return jsonify({"items": [i for i in items if i["id"]]})
        except Exception as exc:  # noqa: BLE001
            return server_error("Impossible de lire Google Drive (est-il connecté ?)", exc)
