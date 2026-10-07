import uuid

import pytest

import app as app_module
import auth
import database


def bearer(token):
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture()
def supa(client, monkeypatch):
    """Enable development bearer authentication."""
    monkeypatch.setattr(auth, "ENABLED", True)
    app_module._known_users.clear()
    return client


def test_valid_bearer_gives_user(supa):
    token = "development-session-token"

    r = supa.get("/api/me", headers=bearer(token))

    assert r.status_code == 200
    assert r.get_json()["auth"] == "supabase"

    # The same bearer token must always map to the same user.
    expected_user = auth.verify_token(token)

    assert database.get_autonomy(expected_user) == "ask"


def test_missing_token_401(supa):
    r = supa.get("/api/me")

    assert r.status_code == 401


def test_invalid_authorization_header_401(supa):
    r = supa.get(
        "/api/me",
        headers={"Authorization": "Basic something"},
    )

    assert r.status_code == 401


def test_empty_bearer_401(supa):
    r = supa.get(
        "/api/me",
        headers={"Authorization": "Bearer "},
    )

    assert r.status_code == 401


def test_same_token_same_user(supa):
    token = "my-development-token"

    user1 = auth.verify_token(token)
    user2 = auth.verify_token(token)

    assert user1 == user2


def test_different_tokens_are_isolated(supa):
    token1 = "user-one-token"
    token2 = "user-two-token"

    user1 = auth.verify_token(token1)
    user2 = auth.verify_token(token2)

    assert user1 != user2

    database.create_user(user1)
    database.create_user(user2)

    aid = database.create_pending_action(
        user1,
        "GMAIL_SEND_EMAIL",
        {},
    )

    # User 2 must not be able to confirm User 1's action.
    response = supa.post(
        f"/api/actions/{aid}/confirm",
        headers=bearer(token2),
    )

    assert response.status_code == 404

    # User 1 can see their own pending action.
    response = supa.get(
        "/api/actions",
        headers=bearer(token1),
    )

    assert response.status_code == 200
    assert response.get_json()["pending_actions"][0]["id"] == aid


def test_public_endpoints_need_no_token(supa):
    response = supa.get("/health")

    assert response.status_code == 200


def test_user_id_is_valid_uuid():
    user_id = auth.verify_token("development-token")

    uuid.UUID(user_id)
