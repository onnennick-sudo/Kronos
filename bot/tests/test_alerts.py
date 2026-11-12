import smtplib

import pytest
import requests

from bot import alerts


EMAIL_ENV = {
    "GMAIL_ADDRESS": "bot@example.invalid",
    "GMAIL_APP_PASSWORD": "app-pass",
    "ALERT_TO_EMAIL": "owner@example.invalid",
}


@pytest.fixture
def clean_env(monkeypatch):
    for k in ("GMAIL_ADDRESS", "GMAIL_APP_PASSWORD", "ALERT_TO_EMAIL", "NTFY_TOPIC"):
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


class FakeSMTP:
    instances = []
    raise_on_send = False

    def __init__(self, host, port, timeout=None):
        self.host, self.port = host, port
        self.logged_in = None
        self.sent = []
        FakeSMTP.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def login(self, user, pw):
        self.logged_in = (user, pw)

    def send_message(self, msg):
        if FakeSMTP.raise_on_send:
            raise smtplib.SMTPException("relay refused")
        self.sent.append(msg)


@pytest.fixture
def fake_smtp(monkeypatch):
    FakeSMTP.instances = []
    FakeSMTP.raise_on_send = False
    monkeypatch.setattr(smtplib, "SMTP_SSL", FakeSMTP)
    return FakeSMTP


def test_send_email_missing_env_returns_false(clean_env, fake_smtp):
    assert alerts.send_email("subj", "body") is False
    assert fake_smtp.instances == []     # never tried to connect


def test_send_email_success(clean_env, fake_smtp):
    for k, v in EMAIL_ENV.items():
        clean_env.setenv(k, v)
    assert alerts.send_email("Hello", "plain body", html="<b>html</b>") is True
    conn = fake_smtp.instances[0]
    assert conn.host == "smtp.gmail.com" and conn.port == 465
    assert conn.logged_in == ("bot@example.invalid", "app-pass")
    msg = conn.sent[0]
    assert msg["Subject"] == "Hello"
    assert msg["To"] == "owner@example.invalid"
    assert msg["From"] == "bot@example.invalid"
    assert msg.get_body(("plain",)).get_content().strip() == "plain body"
    assert msg.get_body(("html",)).get_content().strip() == "<b>html</b>"


def test_send_email_exception_returns_false(clean_env, fake_smtp):
    for k, v in EMAIL_ENV.items():
        clean_env.setenv(k, v)
    fake_smtp.raise_on_send = True
    assert alerts.send_email("Hello", "body") is False


class FakeResponse:
    def __init__(self, status=200):
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


def test_send_push_missing_topic_returns_false(clean_env, monkeypatch):
    calls = []
    monkeypatch.setattr(requests, "post", lambda *a, **k: calls.append((a, k)))
    assert alerts.send_push("t", "m") is False
    assert calls == []


def test_send_push_success_headers_and_url(clean_env, monkeypatch):
    clean_env.setenv("NTFY_TOPIC", "kronos-test-topic")
    captured = {}

    def fake_post(url, data=None, headers=None, timeout=None):
        captured.update(url=url, data=data, headers=headers, timeout=timeout)
        return FakeResponse(200)

    monkeypatch.setattr(requests, "post", fake_post)
    assert alerts.send_push("Halt", "down 2%", priority="urgent",
                            tags=["warning", "bot"]) is True
    assert captured["url"] == "https://ntfy.sh/kronos-test-topic"
    assert captured["data"] == "down 2%".encode("utf-8")
    assert captured["headers"]["Title"] == "Halt"
    assert captured["headers"]["Priority"] == "urgent"
    assert captured["headers"]["Tags"] == "warning,bot"
    assert captured["timeout"] > 0


def test_send_push_invalid_priority_falls_back_to_default(clean_env, monkeypatch):
    clean_env.setenv("NTFY_TOPIC", "t")
    seen = {}
    monkeypatch.setattr(requests, "post",
                        lambda url, data=None, headers=None, timeout=None:
                        seen.update(headers=headers) or FakeResponse(200))
    assert alerts.send_push("x", "y", priority="loud") is True
    assert seen["headers"]["Priority"] == "default"


def test_send_push_http_error_returns_false(clean_env, monkeypatch):
    clean_env.setenv("NTFY_TOPIC", "t")
    monkeypatch.setattr(requests, "post",
                        lambda *a, **k: FakeResponse(500))
    assert alerts.send_push("x", "y") is False


def test_send_push_connection_exception_returns_false(clean_env, monkeypatch):
    clean_env.setenv("NTFY_TOPIC", "t")

    def boom(*a, **k):
        raise requests.ConnectionError("no route")

    monkeypatch.setattr(requests, "post", boom)
    assert alerts.send_push("x", "y") is False


def test_send_push_non_latin1_title_does_not_raise(clean_env, monkeypatch):
    clean_env.setenv("NTFY_TOPIC", "t")
    seen = {}
    monkeypatch.setattr(requests, "post",
                        lambda url, data=None, headers=None, timeout=None:
                        seen.update(headers=headers) or FakeResponse(200))
    assert alerts.send_push("Trade → filled", "m") is True
    assert "→" not in seen["headers"]["Title"]
