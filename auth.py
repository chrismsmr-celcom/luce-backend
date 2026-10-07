"""Authentication.

* If SUPABASE_URL is set: every /api request (except health/callback) must carry
  `Authorization: Bearer <Supabase access token>`. The token is verified here and
  its `sub` claim is the Luce user id. No cookie is involved, so the frontend and
  the API can live on different domains without third-party-cookie problems.
* Otherwise (local development): a signed-cookie anonymous session, as before.
"""
import logging
import os
import uuid

import jwt
from jwt import PyJWKClient

logger = logging.getLogger("luce.auth")

SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
# Legacy projects sign tokens with a shared secret (HS256). Newer projects use
# asymmetric signing keys published at the JWKS endpoint — leave this empty for those.
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "")
SUPABASE_AUDIENCE = os.getenv("SUPABASE_JWT_AUDIENCE", "authenticated")

ENABLED = bool(SUPABASE_URL)

_jwks_client: PyJWKClient | None = None


class AuthError(Exception):
    pass


def _jwks() -> PyJWKClient:
    global _jwks_client
    if _jwks_client is None:
        _jwks_client = PyJWKClient(f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json", cache_keys=True, lifespan=3600)
    return _jwks_client


def verify_token(token: str) -> str:
    """Return the Supabase user id (`sub`) or raise AuthError."""
    try:
        header = jwt.get_unverified_header(token)
        alg = header.get("alg", "")
        if alg == "HS256":
            if not SUPABASE_JWT_SECRET:
                raise AuthError("HS256 token but SUPABASE_JWT_SECRET is not configured")
            key, algorithms = SUPABASE_JWT_SECRET, ["HS256"]
        elif alg in ("ES256", "RS256"):
            key, algorithms = _jwks().get_signing_key_from_jwt(token).key, [alg]
        else:
            raise AuthError("Unsupported token algorithm")  # also rejects alg=none

        claims = jwt.decode(
            token,
            key,
            algorithms=algorithms,
            audience=SUPABASE_AUDIENCE,
            issuer=f"{SUPABASE_URL}/auth/v1",
            options={"require": ["exp", "sub"]},
        )
    except AuthError:
        raise
    except jwt.PyJWTError as exc:
        raise AuthError(f"Invalid token: {type(exc).__name__}") from exc
    except Exception as exc:  # JWKS fetch failures etc.
        logger.warning("Token verification failed: %s", type(exc).__name__)
        raise AuthError("Could not verify token") from exc

    sub = claims["sub"]
    try:
        uuid.UUID(sub)
    except ValueError as exc:
        raise AuthError("Invalid subject") from exc
    return sub


def user_from_authorization(header: str | None) -> str:
    if not header or not header.lower().startswith("bearer "):
        raise AuthError("Missing bearer token")
    return verify_token(header[7:].strip())
