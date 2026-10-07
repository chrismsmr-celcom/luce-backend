"""Simple development authentication.

For development/testing only.

The frontend sends:
    Authorization: Bearer <token>

The backend does NOT decode, verify, or depend on JWT.
The bearer value is treated as an opaque session identifier.

This keeps the API protected from unauthenticated requests while allowing
the frontend and backend to run on different domains without JWT setup.

Replace this with proper authentication before production.
"""

import logging
import uuid

logger = logging.getLogger("luce.auth")

# Authentication is always enabled in this development implementation.
ENABLED = True


class AuthError(Exception):
    """Raised when authentication fails."""


def verify_token(token: str) -> str:
    """Return a stable Luce user id for an opaque bearer token.

    The token is intentionally NOT decoded or verified.

    During development, the token itself is converted into a deterministic
    UUID so the same frontend session maps to the same Luce user.
    """

    token = token.strip()

    if not token:
        raise AuthError("Empty bearer token")

    # Create a deterministic UUID from the opaque token.
    # No JWT parsing or cryptographic verification is performed.
    user_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"luce:{token}"))

    return user_id


def user_from_authorization(header: str | None) -> str:
    """Extract the opaque bearer token and return its Luce user id."""

    if not header:
        raise AuthError("Missing bearer token")

    if not header.lower().startswith("bearer "):
        raise AuthError("Invalid authorization header")

    token = header[7:].strip()

    if not token:
        raise AuthError("Empty bearer token")

    return verify_token(token)
