"""Alerts: email via Gmail SMTP and push via ntfy.sh. Neither function raises."""

from __future__ import annotations

import logging
import os
import smtplib
from email.message import EmailMessage

import requests

log = logging.getLogger(__name__)

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 465
NTFY_BASE_URL = "https://ntfy.sh"
NTFY_TIMEOUT_S = 15
VALID_PRIORITIES = ("default", "high", "urgent")


def send_email(subject: str, text: str, html: str | None = None) -> bool:
    try:
        address = os.environ.get("GMAIL_ADDRESS")
        password = os.environ.get("GMAIL_APP_PASSWORD")
        to_addr = os.environ.get("ALERT_TO_EMAIL")
        missing = [k for k, v in (("GMAIL_ADDRESS", address),
                                  ("GMAIL_APP_PASSWORD", password),
                                  ("ALERT_TO_EMAIL", to_addr)) if not v]
        if missing:
            log.error("send_email skipped: missing env %s", ", ".join(missing))
            return False

        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = address
        msg["To"] = to_addr
        msg.set_content(text)
        if html:
            msg.add_alternative(html, subtype="html")

        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=30) as server:
            server.login(address, password)
            server.send_message(msg)
        return True
    except Exception:  # noqa: BLE001 - contract: never raise
        log.exception("send_email failed")
        return False


def send_push(title: str, message: str, priority: str = "default",
              tags: list[str] | None = None) -> bool:
    try:
        topic = os.environ.get("NTFY_TOPIC")
        if not topic:
            log.error("send_push skipped: NTFY_TOPIC not set")
            return False

        if priority not in VALID_PRIORITIES:
            log.warning("invalid ntfy priority %r; using default", priority)
            priority = "default"

        # HTTP header values must be latin-1; replace anything else so a
        # non-latin-1 title cannot make the request fail.
        safe_title = title.encode("latin-1", "replace").decode("latin-1")
        headers = {"Title": safe_title, "Priority": priority}
        if tags:
            headers["Tags"] = ",".join(tags)

        resp = requests.post(
            f"{NTFY_BASE_URL}/{topic}",
            data=message.encode("utf-8"),
            headers=headers,
            timeout=NTFY_TIMEOUT_S,
        )
        resp.raise_for_status()
        return True
    except Exception:  # noqa: BLE001 - contract: never raise
        log.exception("send_push failed")
        return False
