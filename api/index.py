"""Vercel entrypoint: exposes the Flask WSGI app as a serverless function."""
import os
import sys

# Vercel: filesystem read-only sauf /tmp -> le SDK Composio doit y écrire son cache
os.environ.setdefault("COMPOSIO_CACHE_DIR", "/tmp/.composio")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app  # noqa: E402,F401
