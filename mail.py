"""Lecture complète d'un email (HTML + pièces jointes) et téléchargement des pièces jointes.

Branchement dans app.py (après data.register(...)) :

    import mail  # noqa: E402
    mail.register(app, get_user_id, server_error)

Routes (authentifiées comme le reste de /api) :
    GET /api/data/mail/<message_id>                      -> détail : en-têtes, html, texte, pièces jointes
    GET /api/data/mail/<message_id>/attachment?aid=...   -> octets de la pièce jointe (name=, mime= facultatifs)

Ajoute ?raw=1 pour voir la réponse brute de Composio si un champ n'est pas reconnu.

Limite Vercel : une réponse de fonction serverless ne peut pas dépasser ~4,5 Mo. Au-delà de
LUCE_ATTACHMENT_MAX_BYTES (4 Mo par défaut), la route répond 413 et, si Composio a fourni un lien
de téléchargement temporaire, le renvoie pour que le fichier s'ouvre directement dans un onglet.
"""
import base64
import ipaddress
import logging
import os
import socket
import urllib.error
import urllib.parse
import urllib.request

from flask import Response, jsonify, request

from data import _raw_response, _run, _text

logger = logging.getLogger("luce.mail")

MAX_BYTES = int(os.getenv("LUCE_ATTACHMENT_MAX_BYTES", str(4_000_000)))
MAX_HTML = 1_500_000
MAX_TEXT = 300_000

# Types servis tels quels (le navigateur peut les afficher). Tout le reste est servi en
# application/octet-stream : jamais de HTML / JS servi depuis nos routes.
INLINE_MIME = {
    "image/png", "image/jpeg", "image/jpg", "image/gif", "image/webp", "image/avif", "image/bmp",
    "image/svg+xml", "application/pdf", "video/mp4", "video/webm", "video/quicktime",
    "audio/mpeg", "audio/mp3", "audio/wav", "audio/ogg", "audio/mp4", "audio/webm",
    "text/plain", "text/csv",
}


# ---------------------------------------------------------------------------
# Parsing d'un message
# ---------------------------------------------------------------------------

def _b64(value: str) -> bytes:
    value = value.strip().replace("\n", "")
    value += "=" * (-len(value) % 4)
    try:
        return base64.urlsafe_b64decode(value)
    except Exception:  # noqa: BLE001
        return base64.b64decode(value)


def _find_message(node):
    """Premier dict qui ressemble à un message Gmail."""
    if isinstance(node, dict):
        if any(k in node for k in ("payload", "messageText", "messageId")):
            return node
        for v in node.values():
            found = _find_message(v)
            if found:
                return found
    elif isinstance(node, list):
        for v in node:
            found = _find_message(v)
            if found:
                return found
    return None


def _walk(part, out):
    if isinstance(part, dict):
        out.append(part)
        for sub in part.get("parts") or []:
            _walk(sub, out)


def _headers(payload) -> dict:
    result = {}
    for h in (payload or {}).get("headers") or []:
        if isinstance(h, dict) and h.get("name"):
            result.setdefault(str(h["name"]).lower(), str(h.get("value") or ""))
    return result


def parse_message(data) -> dict:
    msg = _find_message(data) or {}
    payload = msg.get("payload") if isinstance(msg.get("payload"), dict) else {}
    parts: list = []
    _walk(payload, parts)
    hdr = _headers(payload)

    html = text = ""
    attachments: dict[str, dict] = {}

    for p in parts:
        mime = str(p.get("mimeType") or "").lower()
        body = p.get("body") or {}
        filename = str(p.get("filename") or "")
        part_headers = _headers(p)
        content_id = part_headers.get("content-id", "").strip("<> ")
        disposition = part_headers.get("content-disposition", "").lower()

        if body.get("attachmentId") and (filename or content_id):
            attachments[body["attachmentId"]] = {
                "id": body["attachmentId"],
                "filename": filename or f"fichier-{len(attachments) + 1}",
                "mimeType": mime or "application/octet-stream",
                "size": body.get("size"),
                "contentId": content_id,
                "inline": bool(content_id) and not disposition.startswith("attachment"),
            }
        elif body.get("data") and not filename:
            try:
                chunk = _b64(body["data"]).decode("utf-8", "replace")
            except Exception:  # noqa: BLE001
                continue
            if mime == "text/html" and not html:
                html = chunk
            elif mime == "text/plain" and not text:
                text = chunk

    # Format simplifié de Composio (attachmentList) : complète ce que le payload n'a pas donné.
    for a in msg.get("attachmentList") or msg.get("attachments") or []:
        if not isinstance(a, dict):
            continue
        aid = a.get("attachmentId") or a.get("attachment_id") or a.get("id")
        if not aid or aid in attachments:
            continue
        attachments[aid] = {
            "id": aid,
            "filename": _text(a.get("filename") or a.get("name") or "fichier"),
            "mimeType": _text(a.get("mimeType") or a.get("mime_type") or "application/octet-stream"),
            "size": a.get("size"),
            "contentId": "",
            "inline": False,
        }

    html = html or _text(msg.get("messageHtml") or msg.get("html"))
    text = text or _text(msg.get("messageText") or msg.get("message_text") or msg.get("body"))

    def clean_size(s):
        return int(s) if str(s or "").isdigit() else None

    return {
        "id": _text(msg.get("messageId") or msg.get("id")),
        "from": _text(hdr.get("from") or msg.get("sender") or msg.get("from")),
        "to": _text(hdr.get("to") or msg.get("to")),
        "cc": _text(hdr.get("cc") or msg.get("cc")),
        "subject": _text(hdr.get("subject") or msg.get("subject") or "(sans objet)"),
        "date": _text(msg.get("messageTimestamp") or msg.get("date") or hdr.get("date")),
        "html": html[:MAX_HTML],
        "text": text[:MAX_TEXT],
        "attachments": [{**a, "size": clean_size(a["size"])} for a in attachments.values()],
    }


# ---------------------------------------------------------------------------
# Pièce jointe : extraction des octets
# ---------------------------------------------------------------------------

_B64_KEYS = ("data", "file_data", "content", "base64", "attachmentData", "attachment_data", "body")
_URL_KEYS = ("s3url", "s3_url", "url", "download_url", "downloadUrl", "signed_url", "link")


def _find_payload(node, depth=0):
    """Retourne ("b64", str) ou ("url", str) en explorant la réponse."""
    if depth > 6:
        return None
    if isinstance(node, dict):
        for k in _URL_KEYS:
            v = node.get(k)
            if isinstance(v, str) and v.startswith("https://"):
                return ("url", v)
        for k in _B64_KEYS:
            v = node.get(k)
            if isinstance(v, str) and len(v) > 16 and not v.startswith("http"):
                return ("b64", v)
        for v in node.values():
            found = _find_payload(v, depth + 1)
            if found:
                return found
    elif isinstance(node, list):
        for v in node:
            found = _find_payload(v, depth + 1)
            if found:
                return found
    return None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):  # noqa: D401
        return None


def _is_public_https(url: str) -> bool:
    """Garde-fou SSRF : https uniquement, et le nom d'hôte ne doit résoudre que vers des IP publiques."""
    parts = urllib.parse.urlparse(url)
    if parts.scheme != "https" or not parts.hostname:
        return False
    try:
        infos = socket.getaddrinfo(parts.hostname, parts.port or 443, proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        return False
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not ip.is_global:
            return False
    return True


def _download(url: str):
    """Télécharge au plus MAX_BYTES + 1 octets. Retourne (bytes, too_large)."""
    if not _is_public_https(url):
        raise ValueError("URL de téléchargement refusée")
    opener = urllib.request.build_opener(_NoRedirect)
    req = urllib.request.Request(url, headers={"User-Agent": "luce/1.0"})
    with opener.open(req, timeout=20) as resp:  # noqa: S310 (https + IP publique vérifiés)
        declared = int(resp.headers.get("Content-Length") or 0)
        if declared > MAX_BYTES:
            return b"", True
        content = resp.read(MAX_BYTES + 1)
    return content[:MAX_BYTES], len(content) > MAX_BYTES


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

def register(app, get_user_id, server_error):
    @app.get("/api/data/mail/<message_id>")
    def mail_detail(message_id):
        user_id = get_user_id()
        try:
            data, result = _run(
                user_id,
                ["GMAIL_FETCH_MESSAGE_BY_MESSAGE_ID"],
                [{"message_id": message_id, "format": "full"}, {"message_id": message_id}],
            )
            if request.args.get("raw") == "1":
                return _raw_response(result)
            detail = parse_message(data)
            detail["id"] = detail["id"] or message_id
            return jsonify(detail)
        except Exception as exc:  # noqa: BLE001
            return server_error("Impossible de lire ce message", exc)

    @app.get("/api/data/mail/<message_id>/attachment")
    def mail_attachment(message_id):
        user_id = get_user_id()
        attachment_id = request.args.get("aid", "")
        if not attachment_id:
            return jsonify({"error": "Pièce jointe manquante"}), 400
        name = request.args.get("name") or "fichier"
        mime = (request.args.get("mime") or "").lower().split(";")[0].strip()
        try:
            data, result = _run(
                user_id,
                ["GMAIL_GET_ATTACHMENT"],
                [
                    {"message_id": message_id, "attachment_id": attachment_id, "file_name": name},
                    {"message_id": message_id, "attachment_id": attachment_id},
                ],
            )
            if request.args.get("raw") == "1":
                return _raw_response(result)

            found = _find_payload(data)
            if not found:
                raise RuntimeError("Réponse de pièce jointe non reconnue (essaie ?raw=1)")
            kind, value = found

            if kind == "url":
                try:
                    content, too_large = _download(value)
                except ValueError:
                    # URL refusée par le garde-fou SSRF : on ne la renvoie pas au navigateur.
                    logger.warning("URL de pièce jointe refusée (non publique ou non https)")
                    return jsonify({"error": "Pièce jointe indisponible"}), 502
                except (urllib.error.URLError, OSError) as exc:
                    logger.warning("Téléchargement de pièce jointe échoué: %s", type(exc).__name__)
                    return jsonify({"error": "Téléchargement impossible ici", "url": value}), 413
                if too_large:
                    return jsonify({"error": "Fichier trop volumineux pour être affiché ici", "url": value}), 413
            else:
                content = _b64(value)
                if len(content) > MAX_BYTES:
                    return jsonify({"error": "Fichier trop volumineux pour être affiché ici"}), 413

            served = mime if mime in INLINE_MIME else "application/octet-stream"
            safe_name = urllib.parse.quote(name)
            return Response(
                content,
                mimetype=served,
                headers={
                    "Content-Disposition": f"inline; filename*=UTF-8''{safe_name}",
                    "X-Content-Type-Options": "nosniff",
                    "Content-Security-Policy": "default-src 'none'; sandbox",
                },
            )
        except Exception as exc:  # noqa: BLE001
            return server_error("Impossible de récupérer la pièce jointe", exc)
