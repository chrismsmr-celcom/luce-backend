# À ajouter dans luce-backend : nouveau fichier data_snapshots.py
# + une route dans app.py :  register_snapshots(app)
#
# Principe : AUCUN outil prédéfini. On prend les toolkits connectés de
# l'utilisateur (hors messagerie/agenda), on découvre dynamiquement un
# outil de LECTURE chez Composio pour chacun (LIST/FETCH/GET/SEARCH),
# on l'exécute, et on extrait un snapshot {name, headline, items}.
# Chaque outil est isolé dans un try/except : si l'un échoue, les autres
# s'affichent quand même.

import logging
import re

from composio_service import execute_tool, get_or_create_session

logger = logging.getLogger("luce.data_snapshots")

# Outils déjà affichés ailleurs sur la page d'accueil (inbox, agenda, à valider).
EXCLUDED_TOOLKITS = {"gmail", "slack", "googlecalendar"}

# Timeout global par outil (via limite d'items plutôt qu'un vrai timeout).
MAX_ITEMS = 4

# Priorité des verbes de lecture : on cherche un tool "lister les derniers
# éléments" plutôt qu'un tool "lire un élément précis" (qui exige un id).
_READ_PRIORITY = [
    "LIST", "FETCH", "GET", "SEARCH", "FIND", "QUERY",
]


def _pick_read_tool(tools: list[dict], toolkit: str) -> dict | None:
    """Choisit le meilleur outil de lecture d'un toolkit, sans liste en dur.

    Critères :
    - le nom commence par le toolkit (ex. "GOOGLEDRIVE_LIST_FILES")
    - contient un verbe de lecture (préférence à LIST)
    - tous ses paramètres requis sont injectables par défaut
      (sinon on fournit des valeurs par défaut génériques ci-dessous)
    """
    candidates: list[tuple[int, dict]] = []

    for tool in tools:
        fn = tool.get("function", {})
        name = (fn.get("name") or "").upper()

        if not name.startswith(toolkit.upper() + "_"):
            continue

        verb_score = -1
        for i, verb in enumerate(_READ_PRIORITY):
            if f"_{verb}_" in name or name.endswith("_" + verb):
                verb_score = len(_READ_PRIORITY) - i
                break

        if verb_score < 0:
            continue

        # Pénalité si des paramètres requis non devinables existent.
        required = (fn.get("parameters", {}).get("required") or [])
        candidates.append((verb_score - len(required), tool))

    if not candidates:
        return None

    candidates.sort(key=lambda c: c[0], reverse=True)
    return candidates[0][1]


def _default_args(tool: dict) -> dict:
    """Valeurs par défaut génériques pour les paramètres d'un outil de lecture."""
    fn = tool.get("function", {})
    props = fn.get("parameters", {}).get("properties", {}) or {}

    defaults = {
        "limit": MAX_ITEMS,
        "max_results": MAX_ITEMS,
        "count": MAX_ITEMS,
        "page_size": MAX_ITEMS,
        "query": "",          # recherche vide = tout
        "search": "",
        "q": "",
        "sort": "recent",     # ignoré si non supporté
        "order_by": "updated_at",
    }

    return {k: v for k, v in defaults.items() if k in props}


def _extract_items(result) -> list[dict]:
    """Extraction heuristique : trouve la première liste de dicts du résultat,
    puis en tire label + detail à partir de clés usuelles (name, title,
    subject, summary, description, updated_at, created_at, date...)."""
    if isinstance(result, str):
        return []

    def find_list(node):
        if isinstance(node, list) and node and isinstance(node[0], dict):
            return node
        if isinstance(node, dict):
            for value in node.values():
                found = find_list(value)
                if found:
                    return found
        return None

    rows = find_list(result) or []

    label_keys = ["name", "title", "subject", "summary", "display_name", "label"]
    detail_keys = ["updated_at", "created_at", "date", "modified", "status", "owner", "email", "url"]

    items = []
    for row in rows[:MAX_ITEMS]:
        label = next((str(row[k]) for k in label_keys if row.get(k)), "Élément")
        detail = next(
            (str(row[k])[:40] for k in detail_keys if row.get(k) and not isinstance(row.get(k), (dict, list))),
            None,
        )
        items.append({"label": label[:80], "detail": detail})

    return items


_TOOLKIT_LABELS = {
    # uniquement cosmétique ; tout toolkit inconnu est affiché avec son id.
    "googledrive": "Google Drive",
    "github": "GitHub",
    "notion": "Notion",
    "hubspot": "HubSpot",
    "salesforce": "Salesforce",
    "googlesheets": "Google Sheets",
    "googletasks": "Google Tasks",
    "linear": "Linear",
    "jira": "Jira",
    "trello": "Trello",
    "asana": "Asana",
    "airtable": "Airtable",
    "whatsapp": "WhatsApp",
    "telegram": "Telegram",
    "twitter": "X (Twitter)",
    "linkedin": "LinkedIn",
    "stripe": "Stripe",
    "zoom": "Zoom",
    "dropbox": "Dropbox",
    "box": "Box",
    "intercom": "Intercom",
    "zendesk": "Zendesk",
    "calendly": "Calendly",
    "discord": "Discord",
}


def _nice_name(toolkit_id: str) -> str:
    if toolkit_id in _TOOLKIT_LABELS:
        return _TOOLKIT_LABELS[toolkit_id]
    return re.sub(r"[_\-]+", " ", toolkit_id).title()


def build_snapshots(user_id: str) -> list[dict]:
    """Construit un snapshot par toolkit connecté non exclu."""
    session = get_or_create_session(user_id)
    raw_tools = session.tools()

    tools = []
    for tool in raw_tools:
        try:
            fn = tool
            if hasattr(tool, "function") or "function" in (tool or {}):
                fn = tool.get("function", tool) if isinstance(tool, dict) else tool.function
            name = (fn.get("name") if isinstance(fn, dict) else getattr(fn, "name", "")) or ""
            params = (fn.get("parameters") if isinstance(fn, dict) else getattr(fn, "parameters", {})) or {}
            tools.append({"function": {"name": name, "parameters": params,
                                       "description": ""}})
        except Exception:
            continue

    # Toolkits connectés = préfixes des noms de tools.
    toolkit_ids = sorted({
        t["function"]["name"].split("_", 1)[0]
        for t in tools
        if t["function"]["name"]
    })

    snapshots = []

    for toolkit in toolkit_ids:
        if toolkit.lower() in EXCLUDED_TOOLKITS:
            continue

        name = _nice_name(toolkit)

        try:
            tool = _pick_read_tool(tools, toolkit)

            if tool is None:
                snapshots.append({
                    "id": toolkit,
                    "name": name,
                    "headline": "Connecté",
                    "items": [],
                })
                continue

            args = _default_args(tool)
            result = execute_tool(
                user_id=user_id,
                tool_slug=tool["function"]["name"],
                arguments=args,
            )

            items = _extract_items(result)
            headline = (
                f"{len(items)} éléments récents"
                if items else "Connecté — aucun élément récent"
            )

            snapshots.append({
                "id": toolkit,
                "name": name,
                "headline": headline,
                "items": items,
            })

        except Exception as exc:
            logger.warning("Snapshot failed for %s: %s", toolkit, exc)
            # On continue : les autres outils s'affichent malgré l'échec.
            snapshots.append({
                "id": toolkit,
                "name": name,
                "headline": "Connecté (données indisponibles)",
                "items": [],
            })

    return snapshots
