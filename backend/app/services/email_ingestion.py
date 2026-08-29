"""Parses unread Gmail messages (labeled SmashApply) into candidate job links.

Uses imaplib against Gmail's IMAP endpoint. Requires an App Password
(https://myaccount.google.com/apppasswords) since Gmail rejects plain
password IMAP logins for accounts with 2FA enabled.
"""
import email
import imaplib
import re
from dataclasses import dataclass
from email.message import Message
from email.utils import parseaddr

from bs4 import BeautifulSoup

from app.config import get_settings

IMAP_HOST = "imap.gmail.com"
IMAP_PORT = 993

# Links we never want to treat as "the job application link" even if they
# appear in a job-alert email.
LINK_BLOCKLIST_PATTERNS = [
    r"unsubscribe",
    r"mailto:",
    r"tracking",
    r"privacy",
    r"terms-of-service",
    r"help\.",
    r"support\.",
    r"\.gif($|\?)",
    r"\.png($|\?)",
]

# Domains that are very likely to be an actual job posting / ATS link.
LIKELY_JOB_DOMAINS = [
    "greenhouse.io",
    "boards.greenhouse.io",
    "lever.co",
    "myworkdayjobs.com",
    "ashbyhq.com",
    "smartrecruiters.com",
    "icims.com",
    "taleo.net",
    "linkedin.com/jobs",
    "indeed.com/viewjob",
    "indeed.com/rc/clk",
    "workable.com",
    "bamboohr.com",
    "jobvite.com",
    "breezy.hr",
    "recruitee.com",
]


@dataclass
class CandidateJob:
    title: str
    company: str
    raw_url: str
    message_id: str


def _decode_header(raw_value: str | None) -> str:
    if not raw_value:
        return ""
    decoded_parts = email.header.decode_header(raw_value)
    parts = []
    for text, charset in decoded_parts:
        if isinstance(text, bytes):
            parts.append(text.decode(charset or "utf-8", errors="ignore"))
        else:
            parts.append(text)
    return "".join(parts).strip()


def _get_body(msg: Message) -> tuple[str, str]:
    """Returns (html_body, text_body), preferring the richest available part."""
    html_body, text_body = "", ""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            disposition = str(part.get("Content-Disposition") or "")
            if "attachment" in disposition:
                continue
            try:
                payload = part.get_payload(decode=True)
            except Exception:
                continue
            if not payload:
                continue
            charset = part.get_content_charset() or "utf-8"
            text = payload.decode(charset, errors="ignore")
            if content_type == "text/html" and not html_body:
                html_body = text
            elif content_type == "text/plain" and not text_body:
                text_body = text
    else:
        try:
            payload = msg.get_payload(decode=True)
        except Exception:
            payload = None
        if payload:
            charset = msg.get_content_charset() or "utf-8"
            text = payload.decode(charset, errors="ignore")
            if msg.get_content_type() == "text/html":
                html_body = text
            else:
                text_body = text
    return html_body, text_body


def _is_blocked_link(href: str) -> bool:
    return any(re.search(pattern, href, re.IGNORECASE) for pattern in LINK_BLOCKLIST_PATTERNS)


def _is_likely_job_link(href: str) -> bool:
    return any(domain in href.lower() for domain in LIKELY_JOB_DOMAINS)


def extract_links(html_body: str, text_body: str) -> list[str]:
    """Best-effort extraction of plausible job posting links from an email body."""
    candidates: list[str] = []

    if html_body:
        soup = BeautifulSoup(html_body, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            if href.startswith("http") and not _is_blocked_link(href):
                candidates.append(href)
    elif text_body:
        candidates.extend(re.findall(r"https?://[^\s<>\")]+", text_body))
        candidates = [href for href in candidates if not _is_blocked_link(href)]

    # Prefer known ATS/job-board domains, but fall back to whatever survived
    # the blocklist so nothing is dropped on the floor for an unrecognized ATS.
    preferred = [href for href in candidates if _is_likely_job_link(href)]
    ordered = preferred or candidates

    # De-duplicate while preserving order.
    seen: set[str] = set()
    deduped = []
    for href in ordered:
        if href not in seen:
            seen.add(href)
            deduped.append(href)
    return deduped


def _guess_company(from_header: str) -> str:
    _, addr = parseaddr(from_header)
    domain = addr.split("@")[-1] if "@" in addr else ""
    domain = domain.replace("www.", "")
    root = domain.split(".")[0] if domain else ""
    return root.capitalize()


def fetch_new_jobs(mark_as_read: bool = True) -> list[CandidateJob]:
    """Connects to Gmail via IMAP and returns one candidate job per unread
    message in the configured label, each pointing at the first plausible
    application link found in that message.
    """
    settings = get_settings()
    if not settings.gmail_address or not settings.gmail_app_password:
        raise RuntimeError(
            "GMAIL_ADDRESS and GMAIL_APP_PASSWORD must be set in backend/.env "
            "before running Gmail ingestion."
        )

    candidates: list[CandidateJob] = []

    conn = imaplib.IMAP4_SSL(IMAP_HOST, IMAP_PORT)
    try:
        try:
            conn.login(settings.gmail_address, settings.gmail_app_password)
        except imaplib.IMAP4.error as exc:
            raise RuntimeError(
                f"Gmail authentication failed: {exc}. Check GMAIL_ADDRESS and "
                "GMAIL_APP_PASSWORD in backend/.env (an App Password is required "
                "when 2FA is enabled: https://myaccount.google.com/apppasswords)."
            ) from exc

        status, _ = conn.select("INBOX")
        if status != "OK":
            raise RuntimeError("Could not open Gmail INBOX for the configured account.")

        # X-GM-RAW lets us use Gmail's native search syntax (labels, is:unread, etc).
        status, data = conn.search(None, "X-GM-RAW", f'"{settings.gmail_search_query}"')
        if status != "OK":
            raise RuntimeError(
                f"Gmail search failed for query '{settings.gmail_search_query}'. "
                f"Check that the label '{settings.gmail_label}' exists in this Gmail account."
            )

        message_ids = data[0].split()
        for msg_id in message_ids:
            status, msg_data = conn.fetch(msg_id, "(RFC822)")
            if status != "OK" or not msg_data or not msg_data[0]:
                continue

            raw_email = msg_data[0][1]
            msg = email.message_from_bytes(raw_email)

            subject = _decode_header(msg.get("Subject"))
            from_header = _decode_header(msg.get("From"))
            message_id_header = msg.get("Message-ID") or msg_id.decode()

            html_body, text_body = _get_body(msg)
            links = extract_links(html_body, text_body)

            if not links:
                continue

            candidates.append(
                CandidateJob(
                    title=subject or "Untitled role",
                    company=_guess_company(from_header),
                    raw_url=links[0],
                    message_id=message_id_header,
                )
            )

            if mark_as_read:
                conn.store(msg_id, "+FLAGS", "\\Seen")
    finally:
        try:
            conn.logout()
        except Exception:
            pass

    return candidates
