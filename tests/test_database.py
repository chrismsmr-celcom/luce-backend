"""Run the same data-layer tests against SQLite and a real (embedded) Postgres."""
import pytest

import database


@pytest.fixture(scope="session")
def pg_uri(tmp_path_factory):
    pgserver = pytest.importorskip("pgserver")
    server = pgserver.get_server(tmp_path_factory.mktemp("pg"), cleanup_mode="stop")
    yield server.get_uri()
    server.cleanup()


@pytest.fixture(params=["sqlite", "postgres"])
def db(request, tmp_path, monkeypatch):
    if request.param == "sqlite":
        monkeypatch.setattr(database, "USE_PG", False)
        monkeypatch.setattr(database, "DB_PATH", tmp_path / "t.db")
    else:
        uri = request.getfixturevalue("pg_uri")
        import psycopg
        from psycopg.rows import dict_row

        monkeypatch.setattr(database, "USE_PG", True)
        monkeypatch.setattr(database, "DATABASE_URL", uri)
        monkeypatch.setattr(database, "psycopg", psycopg, raising=False)
        monkeypatch.setattr(database, "dict_row", dict_row, raising=False)
        with psycopg.connect(uri, autocommit=True) as c:
            c.execute("DROP SCHEMA public CASCADE; CREATE SCHEMA public;")
    database.init_db()
    database.init_db()  # idempotent
    return request.param


def test_users_and_autonomy(db):
    database.create_user("u1")
    database.create_user("u1")  # duplicate is a no-op
    assert database.get_autonomy("u1") == "ask"
    database.set_autonomy("u1", "auto")
    assert database.get_autonomy("u1") == "auto"
    with pytest.raises(ValueError):
        database.set_autonomy("u1", "root")


def test_messages_limit_and_clear(db):
    database.create_user("u")
    for i in range(10):
        database.save_message("u", "user", f"m{i}")
    assert [m["content"] for m in database.get_messages("u", limit=3)] == ["m7", "m8", "m9"]
    assert len(database.get_messages("u")) == 10
    database.clear_messages("u")
    assert database.get_messages("u") == []


def test_composio_session_upsert(db):
    database.create_user("u")
    assert database.get_composio_session_id("u") is None
    database.save_composio_session_id("u", "v7:a")
    database.save_composio_session_id("u", "v7:b")
    assert database.get_composio_session_id("u") == "v7:b"


def test_pending_actions_lifecycle_and_scoping(db):
    database.create_user("u")
    database.create_user("other")
    aid = database.create_pending_action("u", "GMAIL_SEND_EMAIL", {"to": "a@b.c", "n": 1})
    assert database.get_pending_action("other", aid) is None  # scoped to owner
    got = database.get_pending_action("u", aid)
    assert got["arguments"] == {"to": "a@b.c", "n": 1} and isinstance(got["created_at"], str)
    assert [a["id"] for a in database.list_pending_actions("u")] == [aid]
    assert database.claim_pending_action("u", aid, "confirmed") is True
    assert database.claim_pending_action("u", aid, "confirmed") is False  # no double claim
    assert database.list_pending_actions("u") == []


def test_delete_user_cascades(db):
    database.create_user("u")
    database.save_message("u", "user", "x")
    database.save_composio_session_id("u", "s")
    aid = database.create_pending_action("u", "T", {})
    database.delete_user("u")
    assert database.get_messages("u") == []
    assert database.get_composio_session_id("u") is None
    assert database.get_pending_action("u", aid) is None
