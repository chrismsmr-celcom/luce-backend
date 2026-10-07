import app as app_module
import database


def test_health(client):
    assert client.get("/health").get_json()["status"] == "ok"


def test_post_requires_client_header(client):
    client.environ_base.pop("HTTP_X_LUCE_CLIENT")
    r = client.post("/api/chat", json={"message": "hi"})
    assert r.status_code == 403


def test_foreign_origin_rejected(client):
    r = client.post("/api/chat", json={"message": "hi"}, headers={"Origin": "https://evil.example"})
    assert r.status_code == 403


def test_allowed_origin_passes_csrf(client, monkeypatch):
    monkeypatch.setattr(app_module, "process_message", lambda user_id, message: {"message": "ok"})
    r = client.post("/api/chat", json={"message": "hi"}, headers={"Origin": "https://app.example.com"})
    assert r.status_code == 200


def test_chat_validation(client):
    assert client.post("/api/chat", json={}).status_code == 400
    assert client.post("/api/chat", json={"message": "   "}).status_code == 400
    assert client.post("/api/chat", json={"message": 123}).status_code == 400
    assert client.post("/api/chat", json={"message": "x" * 5000}).status_code == 413


def test_chat_error_does_not_leak_internals(client, monkeypatch):
    def boom(user_id, message):
        raise RuntimeError("secret-api-key-123 leaked")

    monkeypatch.setattr(app_module, "process_message", boom)
    r = client.post("/api/chat", json={"message": "hi"})
    assert r.status_code == 500
    assert "secret" not in r.get_data(as_text=True)


def test_chat_rate_limit(client, monkeypatch):
    monkeypatch.setattr(app_module, "process_message", lambda user_id, message: {"message": "ok"})
    monkeypatch.setattr(app_module, "CHAT_RATE_LIMIT", 2)
    codes = [client.post("/api/chat", json={"message": "hi"}).status_code for _ in range(4)]
    assert codes[:2] == [200, 200] and 429 in codes[2:]


def test_connect_rejects_unknown_toolkit(client):
    assert client.post("/api/connect/not-a-toolkit").status_code == 400


def test_me_lists_toolkits_and_defaults_to_ask(client):
    data = client.get("/api/me").get_json()
    assert "gmail" in data["toolkits"] and "slack" in data["toolkits"]
    assert data["autonomy"] == "ask"


def test_settings_autonomy(client):
    assert client.post("/api/settings", json={"autonomy": "root"}).status_code == 400
    assert client.post("/api/settings", json={"autonomy": "draft"}).status_code == 200
    assert client.get("/api/me").get_json()["autonomy"] == "draft"


def test_pending_action_confirm_once_and_user_scoped(client, monkeypatch):
    client.get("/api/me")  # create the session/user
    with client.session_transaction() as s:
        uid = s["user_id"]
    action_id = database.create_pending_action(uid, "GMAIL_SEND_EMAIL", {"to": "a@b.c"})

    calls = []
    monkeypatch.setattr(
        app_module, "run_confirmed_action",
        lambda user_id, tool, args: calls.append((tool, args)) or {"success": True},
    )

    # Another browser (another user) cannot touch it.
    other = app_module.app.test_client()
    other.environ_base["HTTP_X_LUCE_CLIENT"] = "web"
    assert other.post(f"/api/actions/{action_id}/confirm").status_code == 404

    assert client.post(f"/api/actions/{action_id}/confirm").status_code == 200
    assert client.post(f"/api/actions/{action_id}/confirm").status_code == 409  # no double execution
    assert calls == [("GMAIL_SEND_EMAIL", {"to": "a@b.c"})]


def test_reject_action(client):
    client.get("/api/me")
    with client.session_transaction() as s:
        uid = s["user_id"]
    action_id = database.create_pending_action(uid, "GMAIL_SEND_EMAIL", {})
    assert client.post(f"/api/actions/{action_id}/reject").status_code == 200
    assert client.get("/api/actions").get_json()["pending_actions"] == []


def test_account_delete_removes_data(client):
    client.get("/api/me")
    with client.session_transaction() as s:
        uid = s["user_id"]
    database.save_message(uid, "user", "hello")
    assert client.post("/api/account/delete").status_code == 200
    assert database.get_messages(uid) == []
