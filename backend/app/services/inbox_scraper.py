"""Desktop Gmail OAuth and referral extraction for the local, single-user app.

Fetched messages are marked read before returning, including malformed bodies.
"""
import base64
import json
import re
from email.header import decode_header, make_header
from email.message import Message
from email.utils import parseaddr
from pathlib import Path
from threading import Lock

from bs4 import BeautifulSoup

from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

ROOT = Path(__file__).resolve().parents[3]
SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]
QUERY = 'is:unread (referral OR "job description" OR opportunity OR role)'
WORKFLOW_LOCK = Lock()


class InboxError(Exception):
    """A safe, user-facing Gmail error."""


def _write_json(path: Path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value), encoding="utf-8")
    temporary.replace(path)


def _gmail_service():
    token = ROOT / "token.json"
    credentials = None
    try:
        if token.exists():
            try:
                credentials = Credentials.from_authorized_user_file(str(token))
            except ValueError:
                credentials = None
            if credentials and not credentials.has_scopes(SCOPES):
                credentials = None
        if credentials and credentials.expired and credentials.refresh_token:
            try:
                credentials.refresh(Request())
            except RefreshError:
                credentials = None
        if not credentials or not credentials.valid:
            secrets = ROOT / "credentials.json"
            if not secrets.exists():
                secrets = ROOT / "credentials.json.json"
            if not secrets.exists():
                raise InboxError("Place Google Desktop OAuth credentials.json in the project root, then retry.")
            flow = InstalledAppFlow.from_client_secrets_file(str(secrets), SCOPES)
            credentials = flow.run_local_server(port=0, timeout_seconds=180)
        _write_json(token, json.loads(credentials.to_json()))
        return build("gmail", "v1", credentials=credentials, cache_discovery=False)
    except InboxError:
        raise
    except Exception as exc:
        raise InboxError("Gmail authentication failed. Check Desktop OAuth credentials and complete consent in the browser on the backend computer, then retry.") from exc


def extract_body(payload, *, service=None, message_id=None):
    """Extract MIME body text, preferring plain text within alternatives.

    Ignore file attachments; Gmail attachment-backed inline bodies are fetched
    when a service is provided. Nested mixed bodies retain their text sections.
    """
    if not isinstance(payload, dict):
        return ""
    mime = payload.get("mimeType", "").lower()
    headers = payload.get("headers") or []
    disposition = next((h.get("value", "") for h in headers
                        if h.get("name", "").lower() == "content-disposition"), "")
    if payload.get("filename") or disposition.lower().startswith("attachment") or mime == "message/rfc822":
        return ""
    parts = payload.get("parts") or []
    if parts:
        if mime == "multipart/alternative":
            # Plain text can occur after HTML in MIME order.
            parts = sorted(parts, key=lambda part: part.get("mimeType") != "text/plain")
            for part in parts:
                text = extract_body(part, service=service, message_id=message_id)
                if text.strip():
                    return text
            return ""
        return "\n\n".join(text for part in parts
                           if (text := extract_body(part, service=service, message_id=message_id)).strip())
    if mime not in ("text/plain", "text/html", ""):
        return ""
    body = payload.get("body") or {}
    data = body.get("data")
    if not data and body.get("attachmentId") and service is not None:
        data = service.users().messages().attachments().get(
            userId="me", messageId=message_id, id=body["attachmentId"]
        ).execute().get("data")
    if not data:
        return ""
    try:
        raw = base64.b64decode(data + "=" * (-len(data) % 4), altchars=b"-_", validate=True)
    except (ValueError, TypeError):
        return ""
    header = Message()
    header["content-type"] = next((h.get("value", "") for h in headers
                                    if h.get("name", "").lower() == "content-type"), mime)
    try:
        decoded = raw.decode(header.get_content_charset() or "utf-8", errors="replace")
    except LookupError:
        decoded = raw.decode("utf-8", errors="replace")
    if mime == "text/html":
        soup = BeautifulSoup(decoded, "html.parser")
        for element in soup(["script", "style", "head"]):
            element.decompose()
        return soup.get_text(separator="\n", strip=True)
    return decoded.strip()


_FIELDS = re.compile(
    r"(?im)^[ \t]*(?P<label>Job\s+Title|Duration|Job\s+Summary|Job\s+Description)\s*:+[ \t]*"
)


def parse_referral(body: str, headers: list[dict]) -> dict[str, str]:
    """Parse recruiter fields locally, keeping the JD heading through EOF.

    Unstructured messages retain their body; absent titles are explicitly unknown.
    Field values may be on the next line, as often happens with HTML tables.
    """
    body = body.replace("\r\n", "\n").replace("\r", "\n")
    fields = list(_FIELDS.finditer(body))
    values = {}
    jd_start = None
    for index, match in enumerate(fields):
        label = " ".join(match["label"].lower().split())
        end = fields[index + 1].start() if index + 1 < len(fields) else len(body)
        value = body[match.end():end].strip()
        if label in ("job summary", "job description"):
            if jd_start is None:
                jd_start = match.start()
        elif label not in values and jd_start is None:
            values[label] = value.splitlines()[0].strip() if value else ""
    sender_header = next((h.get("value", "") for h in headers
                          if h.get("name", "").lower() == "from"), "")
    try:
        sender_header = str(make_header(decode_header(sender_header)))
    except (ValueError, LookupError):
        pass
    name, address = parseaddr(sender_header)
    sender = f"{name} <{address}>" if name and address else name or address or "Unknown sender"
    return {
        "job_title": values.get("job title") or "Unknown job title",
        "sender": sender,
        "duration": values.get("duration", ""),
        "jd_text": body[jd_start:].strip() if jd_start is not None else body.strip(),
    }


def fetch_latest_referral_jd() -> dict[str, str] | None:
    """Fetch one unread referral and mark it read even when extraction fails."""
    service = _gmail_service()
    try:
        page = service.users().messages().list(
            userId="me", q=QUERY, maxResults=1, labelIds=["INBOX"]
        ).execute()
        messages = page.get("messages", [])
        if not messages:
            return None
        message_id = messages[0]["id"]
        message = service.users().messages().get(
            userId="me", id=message_id, format="full"
        ).execute()
        try:
            payload = message.get("payload", {})
            body = extract_body(payload, service=service, message_id=message_id)
            return parse_referral(body, payload.get("headers") or [])
        finally:
            # Only claim the message was read if Gmail confirms this mutation.
            try:
                service.users().messages().modify(
                    userId="me", id=message_id, body={"removeLabelIds": ["UNREAD"]}
                ).execute()
            except HttpError as exc:
                raise InboxError("Could not mark the referral email as read. Check Gmail modify access and retry.") from exc
    except HttpError as exc:
        raise InboxError("Gmail could not fetch the referral. Check that the Gmail API is enabled and account access is authorized, then retry.") from exc
    finally:
        service.close()
