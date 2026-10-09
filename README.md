# Luce — backend

API Flask de **Luce**, l'assistant qui lit tes outils (Gmail, Agenda, Drive…), comprend ce qui compte et
**prépare le travail avant que tu le demandes** : réponses rédigées, résumé de la journée, rappels, alertes.
Chaque action passe par ta validation et par la couche de sécurité **Cerbère**.

> Frontend : [Luce](https://github.com/chrismsmr-celcom/Luce) (TanStack Start + React).

## Ce que fait le backend

| Domaine | Détail |
|---|---|
| **Connexions** | OAuth des outils via [Composio](https://composio.dev) (`/api/connect/<outil>`, `/api/connections`) |
| **Lecture** | Inbox Gmail, agenda du jour, fichiers Drive, détail d'un mail (HTML + pièces jointes) |
| **Artefacts proactifs** | `proposals.py` lit tes outils, un seul appel au modèle prépare des propositions (réponses, résumé, rappels, alertes) |
| **Actions** | Validation en un clic → exécution via `run_confirmed_action` (Cerbère). Une réponse devient un **brouillon Gmail**, jamais un envoi automatique |
| **Chat** | Agent à outils (`/api/chat`) avec 3 niveaux d'autonomie : `ask`, `draft`, `auto` |
| **Sécurité** | Auth Supabase (JWT vérifié), CSRF, CORS strict, limitation de débit, sandbox des contenus |

## Architecture

```
app.py            routes principales, CORS, CSRF, rate limiting
auth.py           vérification du jeton Supabase (sub = identifiant utilisateur stable)
agent.py          boucle agent (chat) + exécution des actions confirmées
composio_service.py   sessions Composio, OAuth, execute_tool
cerbere_service.py    garde-fou Cerbère / AgentGuard sur chaque action externe
data.py           lecture Gmail / Agenda / Drive (/api/data/*)
mail.py           détail d'un mail + proxy des pièces jointes (/api/data/mail/*)
proposals.py      moteur d'artefacts proactifs (/api/artifacts/*)
database.py       SQLite en local, PostgreSQL (Supabase) en production
toolkits.py       liste unique des outils connectables
api/index.py      point d'entrée Vercel
```

### Comment naissent les artefacts

1. Le serveur **lit lui-même** les outils connectés (mêmes appels que l'Inbox/Agenda/Dossiers).
2. **Un seul appel** au modèle reçoit ces données (marquées *non fiables*) et rend du JSON.
3. Le modèle n'a **aucun outil** : un mail piégé ne peut rien exécuter.
4. Une action n'est jamais décidée par le modèle : le serveur la construit (destinataire = expéditeur réel du
   mail, action = brouillon Gmail).
5. L'exécution n'a lieu qu'au clic sur « Valider ».

Pour brancher un nouvel outil (GitHub, CRM…), ajoute un collecteur dans `COLLECTORS` (`proposals.py`).

## Démarrage local

```sh
python -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt
# crée un fichier .env avec les variables ci-dessous
python app.py               # http://127.0.0.1:10000
pytest
```

## Variables d'environnement

**Obligatoires**

| Variable | Rôle |
|---|---|
| `FLASK_SECRET_KEY` | Clé de signature des sessions |
| `COMPOSIO_API_KEY` | Accès Composio |
| `FRONTEND_ORIGINS` | Origine(s) autorisée(s) du frontend, ex. `https://luce-phi.vercel.app` (sans chemin). Sans elle, le navigateur bloque tout (CORS) |
| `SUPABASE_URL` | Active la vérification des jetons (`https://xxxx.supabase.co`) |
| `LUCE_ENV` | `production` en production (cookies sécurisés) |

**Selon ton projet Supabase**

| Variable | Rôle |
|---|---|
| `SUPABASE_JWT_SECRET` | Projets qui signent encore en HS256 (Project Settings → API → JWT secret). Inutile avec les clés asymétriques (JWKS) |
| `SUPABASE_JWT_AUDIENCE` | Défaut `authenticated` |
| `DATABASE_URL` / `POSTGRES_URL` | PostgreSQL en production. Sans elles : SQLite (`LUCE_DB_PATH`) |

**Utiles**

| Variable | Défaut | Rôle |
|---|---|---|
| `FRONTEND_URL` | `/connexions` | Où revenir après l'OAuth Composio — mets l'URL complète du frontend |
| `PUBLIC_BASE_URL` | — | URL publique du backend (callback OAuth) |
| `LUCE_PROVIDERS` | — | Ordre/liste des fournisseurs de modèle (DeepSeek, OpenRouter, Cerebras) |
| `LUCE_ATTACHMENT_MAX_BYTES` | `4000000` | Taille max d'une pièce jointe servie (limite Vercel ≈ 4,5 Mo) |
| `LUCE_PROPOSAL_MAX_EMAILS` | `12` | Mails analysés par passe |
| `LUCE_CHAT_RATE_LIMIT` / `LUCE_CHAT_RATE_WINDOW` | `20` / `60` | Limitation de débit du chat |
| `AGENTGUARD_API_KEY`, `AGENTGUARD_COLLECTOR_URL` | — | Cerbère / AgentGuard |

## Déploiement (Vercel)

1. Importer ce dépôt comme projet Vercel (Python). `vercel.json` fixe la durée max des fonctions.
2. Renseigner les variables ci-dessus (cochées pour **Production**), puis **redéployer**.
3. Vérifier `https://<backend>/api/health` : `cors_origins` doit contenir l'URL du frontend.

## Sécurité en bref

- Jeton Supabase **vérifié** (signature, issuer, audience, expiration) ; `sub` = identité stable.
- Les contenus de mails s'affichent dans une iframe sans script, avec CSP ; images distantes bloquées par défaut.
- Les pièces jointes sont servies avec un type restreint ; téléchargement serveur protégé contre le SSRF.
- Aucune action d'écriture sans validation (niveau `ask` par défaut) ; tout passe par Cerbère.

## Endpoints

`GET /api/health` · `GET /api/me` · `POST /api/connect/<outil>` · `GET /api/connections` ·
`GET /api/data/inbox|agenda|files` · `GET /api/data/mail/<id>` · `GET /api/data/mail/<id>/attachment` ·
`GET /api/artifacts` · `POST /api/artifacts/generate|<id>/approve|<id>/dismiss` ·
`POST /api/chat` · `GET /api/actions` · `POST /api/actions/<id>/confirm|reject` · `POST /api/account/delete`

Ajoute `?raw=1` aux routes `/api/data/*` pour voir la réponse brute de Composio (débogage).

## Feuille de route

Lecture GitHub, Slack et CRM · tâche planifiée (briefing du matin) · mémoire des préférences ·
réponses envoyées après validation · résumé hebdomadaire.
