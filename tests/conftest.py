import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("FLASK_SECRET_KEY", "test-secret")
os.environ.setdefault("COMPOSIO_API_KEY", "test")
os.environ.setdefault("AGENTGUARD_API_KEY", "test")
os.environ.setdefault("DEEPSEEK_API_KEY", "test")
os.environ.setdefault("FRONTEND_ORIGINS", "https://app.example.com")

import pytest  # noqa: E402


@pytest.fixture()
def client(tmp_path, monkeypatch):
    import database

    monkeypatch.setattr(database, "DB_PATH", tmp_path / "test.db")
    database.init_db()
    import app as app_module

    app_module._hits.clear()
    app_module.app.config["TESTING"] = True
    c = app_module.app.test_client()
    c.environ_base["HTTP_X_LUCE_CLIENT"] = "web"
    return c
