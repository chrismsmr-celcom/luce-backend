import agent
import database


def test_write_tool_classification():
    assert agent.is_write_tool("GMAIL_SEND_EMAIL")
    assert agent.is_write_tool("GOOGLEDRIVE_DELETE_FILE")
    assert agent.is_write_tool("GMAIL_CREATE_EMAIL_DRAFT")
    assert agent.is_write_tool("SOMETHING_UNKNOWN_THING")  # unknown verbs fail safe
    assert not agent.is_write_tool("GMAIL_FETCH_EMAILS")
    assert not agent.is_write_tool("GOOGLECALENDAR_LIST_EVENTS")


def test_autonomy_policy():
    assert agent.requires_confirmation("GMAIL_SEND_EMAIL", "ask")
    assert agent.requires_confirmation("GMAIL_SEND_EMAIL", "draft")
    assert not agent.requires_confirmation("GMAIL_CREATE_EMAIL_DRAFT", "draft")
    assert agent.requires_confirmation("GMAIL_CREATE_EMAIL_DRAFT", "ask")
    assert not agent.requires_confirmation("GMAIL_SEND_EMAIL", "auto")
    assert not agent.requires_confirmation("GMAIL_FETCH_EMAILS", "ask")


def test_history_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "t.db")
    database.init_db()
    database.create_user("u")
    for i in range(10):
        database.save_message("u", "user", f"m{i}")
    got = database.get_messages("u", limit=3)
    assert [m["content"] for m in got] == ["m7", "m8", "m9"]


class _Fn:
    def __init__(self, name, args):
        self.name, self.arguments = name, args


class _TC:
    id = "1"

    def __init__(self, name, args="{}"):
        self.function = _Fn(name, args)


class _Msg:
    def __init__(self, content=None, tool_calls=None):
        self.content, self.tool_calls = content, tool_calls


class _Resp:
    def __init__(self, msg):
        self.choices = [type("C", (), {"message": msg})]


def test_write_tool_is_queued_not_executed_in_ask_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "t.db")
    database.init_db()
    database.create_user("u")

    executed = []
    monkeypatch.setattr(agent, "execute_with_cerbere", lambda *a, **k: executed.append(a) or {"success": True})
    monkeypatch.setattr(agent, "get_or_create_session", lambda uid: None)
    monkeypatch.setattr(agent, "get_composio_tools", lambda uid: [])

    replies = iter([
        _Resp(_Msg(tool_calls=[_TC("GMAIL_SEND_EMAIL", '{"to": "x@y.z"}')])),
        _Resp(_Msg(content="C'est en attente de ta confirmation.")),
    ])
    monkeypatch.setattr(agent, "call_llm", lambda messages, tools=None: next(replies))

    out = agent.process_message("u", "envoie un mail")
    assert executed == []
    assert len(out["pending_actions"]) == 1
    assert out["pending_actions"][0]["tool"] == "GMAIL_SEND_EMAIL"
    assert len(database.list_pending_actions("u")) == 1


def test_read_tool_runs_directly(tmp_path, monkeypatch):
    monkeypatch.setattr(database, "DB_PATH", tmp_path / "t.db")
    database.init_db()
    database.create_user("u")

    executed = []
    monkeypatch.setattr(agent, "execute_with_cerbere", lambda uid, tool, args: executed.append(tool) or {"success": True})
    monkeypatch.setattr(agent, "get_or_create_session", lambda uid: None)
    monkeypatch.setattr(agent, "get_composio_tools", lambda uid: [])
    replies = iter([
        _Resp(_Msg(tool_calls=[_TC("GMAIL_FETCH_EMAILS")])),
        _Resp(_Msg(content="Voici tes mails.")),
    ])
    monkeypatch.setattr(agent, "call_llm", lambda messages, tools=None: next(replies))

    out = agent.process_message("u", "mes mails")
    assert executed == ["GMAIL_FETCH_EMAILS"]
    assert out["message"] == "Voici tes mails."
