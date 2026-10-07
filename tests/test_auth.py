import time
import uuid

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import ec

import app as app_module
import auth
import database

URL = "https://proj.supabase.co"
SECRET = "super-secret-jwt-key-for-tests-32bytes!"
USER = str(uuid.uuid4())


def make_token(key=SECRET, alg="HS256", **over):
    claims = {
        "sub": USER,
        "aud": "authenticated",
        "iss": f"{URL}/auth/v1",
        "exp": int(time.time()) + 3600,
        **over,
    }
    return jwt.encode({k: v for k, v in claims.items() if v is not None}, key, algorithm=alg)


@pytest.fixture()
def supa(client, monkeypatch):
    monkeypatch.setattr(auth, "ENABLED", True)
    monkeypatch.setattr(auth, "SUPABASE_URL", URL)
    monkeypatch.setattr(auth, "SUPABASE_JWT_SECRET", SECRET)
    app_module._known_users.clear()
    return client


def bearer(token):
    return {"Authorization": f"Bearer {token}"}


def test_valid_token_gives_user(supa):
    r = supa.get("/api/me", headers=bearer(make_token()))
    assert r.status_code == 200 and r.get_json()["auth"] == "supabase"
    assert database.get_autonomy(USER) == "ask"  # user row was created


def test_missing_token_401(supa):
    assert supa.get("/api/me").status_code == 401


@pytest.mark.parametrize(
    "over",
    [{"exp": int(time.time()) - 10}, {"aud": "other"}, {"iss": "https://evil.example/auth/v1"}, {"sub": "not-a-uuid"}],
)
def test_bad_claims_rejected(supa, over):
    assert supa.get("/api/me", headers=bearer(make_token(**over))).status_code == 401


def test_wrong_signature_and_alg_none_rejected(supa):
    assert supa.get("/api/me", headers=bearer(make_token(key="x" * 40))).status_code == 401
    none_token = jwt.encode({"sub": USER, "aud": "authenticated"}, None, algorithm="none")
    assert supa.get("/api/me", headers=bearer(none_token)).status_code == 401


def test_asymmetric_token_via_jwks(supa, monkeypatch):
    key = ec.generate_private_key(ec.SECP256R1())

    class FakeJWKS:
        def get_signing_key_from_jwt(self, token):
            return type("K", (), {"key": key.public_key()})

    monkeypatch.setattr(auth, "_jwks_client", FakeJWKS())
    token = make_token(key=key, alg="ES256")
    assert supa.get("/api/me", headers=bearer(token)).status_code == 200


def test_users_are_isolated(supa):
    t1, other = make_token(), str(uuid.uuid4())
    t2 = make_token(sub=other)
    database.create_user(USER)
    aid = database.create_pending_action(USER, "GMAIL_SEND_EMAIL", {})
    assert supa.post(f"/api/actions/{aid}/confirm", headers=bearer(t2)).status_code == 404
    assert supa.get("/api/actions", headers=bearer(t1)).get_json()["pending_actions"][0]["id"] == aid


def test_public_endpoints_need_no_token(supa):
    assert supa.get("/health").status_code == 200
