import base64
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import HTTPException
from app.services import inbox_scraper as inbox
from app.routers import inbox as router
from app.services import tailor


def encoded(text, encoding="utf-8"):
    return base64.urlsafe_b64encode(text.encode(encoding)).decode().rstrip("=")


class InboxTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = patch.object(inbox, "ROOT", Path(self.temp.name))
        root.start()
        self.addCleanup(root.stop)
        self.service = MagicMock()
        service = patch.object(inbox, "_gmail_service", return_value=self.service)
        service.start()
        self.addCleanup(service.stop)
        self.service.users().getProfile().execute.return_value = {"emailAddress": "test@example.com"}
        self.messages = self.service.users().messages()
        self.messages.list().execute.return_value = {"messages": [{"id": "new"}]}
        self.messages.get().execute.return_value = {"payload": {
            "mimeType": "multipart/mixed", "parts": [
                {"mimeType": "multipart/alternative", "parts": [
                    {"mimeType": "text/html", "body": {"data": encoded("<p>Wrong</p>")}},
                    {"mimeType": "text/plain", "body": {"data": encoded("Role: Engineer")}},
                ]},
                {"mimeType": "text/plain", "filename": "notes.txt", "body": {"data": encoded("Wrong attachment")}},
            ]}}

    def test_nested_plain_text_is_preferred_and_marked_read(self):
        self.assertEqual(inbox.fetch_latest_referral_jd()["jd_text"], "Role: Engineer")
        self.messages.list.assert_called_with(userId="me", q=inbox.QUERY, maxResults=1, labelIds=["INBOX"])
        self.messages.modify.assert_called_once_with(userId="me", id="new", body={"removeLabelIds": ["UNREAD"]})

    def test_empty_inbox_does_not_modify_messages(self):
        self.messages.list().execute.return_value = {}
        self.assertIsNone(inbox.fetch_latest_referral_jd())
        self.messages.modify.assert_not_called()

    def test_html_nested_html_and_malformed_body_are_marked_read(self):
        for payload, expected in [
            ({"mimeType": "text/html", "body": {"data": encoded("<style>hidden</style><p>Role &amp; scope</p><p>Engineer</p>")}}, "Role & scope\nEngineer"),
            ({"mimeType": "multipart/mixed", "parts": [{"mimeType": "multipart/alternative", "parts": [
                {"mimeType": "text/plain", "body": {"data": encoded(" ")}},
                {"mimeType": "text/html", "body": {"data": encoded("<p>Engineer</p>")}}
            ]}]}, "Engineer"),
            ({"mimeType": "text/plain", "body": {"data": "a"}}, ""),
            ({}, ""),
        ]:
            with self.subTest(payload=payload):
                self.messages.modify.reset_mock()
                self.messages.get().execute.return_value = {"payload": payload}
                self.assertEqual(inbox.fetch_latest_referral_jd()["jd_text"], expected)
                self.messages.modify.assert_called_once()

    def test_extraction_exception_still_marks_read(self):
        with patch.object(inbox, "extract_body", side_effect=RuntimeError("parse failed")):
            with self.assertRaises(RuntimeError):
                inbox.fetch_latest_referral_jd()
        self.messages.modify.assert_called_once()
        self.service.close.assert_called_once()

    def test_attachment_backed_body_and_file_attachment_exclusion(self):
        self.messages.attachments().get().execute.return_value = {"data": encoded("R\u00f4le", "iso-8859-1")}
        part = {"mimeType": "text/plain", "headers": [{"name": "Content-Type", "value": "text/plain; charset=iso-8859-1"}], "body": {"attachmentId": "body"}}
        self.assertEqual(inbox.extract_body(part, service=self.service, message_id="new"), "R\u00f4le")
        part["headers"].append({"name": "Content-Disposition", "value": "attachment"})
        self.assertEqual(inbox.extract_body(part, service=self.service, message_id="new"), "")

    def test_mark_read_failure_is_reported(self):
        self.messages.modify().execute.side_effect = inbox.HttpError(
            MagicMock(status=403, reason="Forbidden"), b'{}')
        with self.assertRaisesRegex(inbox.InboxError, "Could not mark"):
            inbox.fetch_latest_referral_jd()


class OAuthTests(unittest.TestCase):
    def test_valid_token_is_reused_and_expired_token_is_refreshed(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(inbox, "ROOT", Path(directory)), \
                patch.object(inbox, "Credentials") as credentials, \
                patch.object(inbox, "InstalledAppFlow") as flow, patch.object(inbox, "build"):
            (Path(directory) / "token.json").write_text("{}")
            token = credentials.from_authorized_user_file.return_value
            token.to_json.return_value = "{}"
            token.expired = False
            token.valid = True
            inbox._gmail_service()
            flow.from_client_secrets_file.assert_not_called()
            token.refresh.assert_not_called()
            token.expired = True
            inbox._gmail_service()
            token.refresh.assert_called_once()

    def test_first_use_saves_token_with_modify_scope(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(inbox, "ROOT", Path(directory)), \
                patch.object(inbox, "InstalledAppFlow") as flow, patch.object(inbox, "build"):
            secrets = Path(directory) / "credentials.json"
            secrets.write_text("{}")
            flow.from_client_secrets_file().run_local_server().to_json.return_value = '{"refresh_token":"test"}'
            inbox._gmail_service()
            flow.from_client_secrets_file.assert_called_with(str(secrets), inbox.SCOPES)
            self.assertTrue((Path(directory) / "token.json").exists())

    def test_readonly_token_triggers_new_consent(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(inbox, "ROOT", Path(directory)), \
                patch.object(inbox, "Credentials") as credentials, \
                patch.object(inbox, "InstalledAppFlow") as flow, patch.object(inbox, "build"):
            (Path(directory) / "token.json").write_text("{}")
            (Path(directory) / "credentials.json").write_text("{}")
            credentials.from_authorized_user_file.return_value.has_scopes.return_value = False
            flow.from_client_secrets_file().run_local_server().to_json.return_value = "{}"
            inbox._gmail_service()
            flow.from_client_secrets_file.assert_called_with(str(Path(directory) / "credentials.json"),
                                                           ["https://www.googleapis.com/auth/gmail.modify"])


class InboxRouteTests(unittest.IsolatedAsyncioTestCase):
    async def test_success_returns_pdf(self):
        referral = {"job_title": "Engineer", "sender": "Recruiter", "jd_text": "Role: Engineer. Build reliable systems and maintain cloud infrastructure."}
        with patch.object(router, "_get_master_cv_or_400", return_value="master"), \
                patch.object(router, "fetch_latest_referral_jd", return_value=referral), \
                patch.object(router, "generate_tailored_resume", new=AsyncMock(return_value=b"%PDF-test")) as generate:
            response = await router.tailor_latest_email(MagicMock())
            self.assertEqual(response.body, b"%PDF-test")
            self.assertEqual(response.media_type, "application/pdf")
            generate.assert_awaited_once_with(referral["jd_text"], "master", job_title="Engineer")

    async def test_failed_generation_explains_how_to_retry(self):
        with patch.object(router, "_get_master_cv_or_400"), \
                patch.object(router, "fetch_latest_referral_jd", return_value={"job_title": "Engineer", "sender": "Recruiter", "jd_text": "Engineer " * 10}), \
                patch.object(router, "generate_tailored_resume", new=AsyncMock(side_effect=router.TailoringError("failed"))):
            with self.assertRaises(HTTPException) as caught:
                await router.tailor_latest_email(MagicMock())
            self.assertEqual(caught.exception.status_code, 502)
            self.assertIn("Mark it unread", caught.exception.detail)
            self.assertFalse(inbox.WORKFLOW_LOCK.locked())

    async def test_invalid_jd_returns_422_without_tailoring(self):
        for text in (None, "", "   ", "x" * 49):
            with self.subTest(text=text), patch.object(router, "_get_master_cv_or_400"), \
                    patch.object(router, "fetch_latest_referral_jd", return_value=None if text is None else {"job_title": "Engineer", "sender": "Recruiter", "jd_text": text}), \
                    patch.object(router, "generate_tailored_resume", new=AsyncMock()) as generate:
                response = await router.tailor_latest_email(MagicMock())
                self.assertEqual(response.status_code, 422)
                detail = json.loads(response.body)["detail"]
                self.assertIn("No unread" if text is None else "Marked as read", detail)
                generate.assert_not_called()
                self.assertFalse(inbox.WORKFLOW_LOCK.locked())

    async def test_concurrent_request(self):
        with patch.object(router, "_get_master_cv_or_400"):
            inbox.WORKFLOW_LOCK.acquire()
            try:
                with self.assertRaises(HTTPException) as caught:
                    await router.tailor_latest_email(MagicMock())
                self.assertEqual(caught.exception.status_code, 409)
            finally:
                inbox.WORKFLOW_LOCK.release()


class StructuredReferralTests(unittest.TestCase):
    def test_fields_and_jd_to_end(self):
        body = "Hello candidate\nJob Title :: Cloud Engineer\nDuration :: 12 months\nJob Summary:\nBuild cloud systems.\nRegards, Recruiter"
        result = inbox.parse_referral(body, [{"name": "From", "value": "Recruiter <jobs@example.com>"}])
        self.assertEqual(result, {"job_title": "Cloud Engineer", "duration": "12 months",
            "sender": "Recruiter <jobs@example.com>", "jd_text": "Job Summary:\nBuild cloud systems.\nRegards, Recruiter"})

    def test_html_table_split_fields_case_and_encoded_sender(self):
        html = "<table><tr><td>JOB TITLE</td><td>::</td><td>Developer</td></tr><tr><td>Duration ::</td><td>6 months</td></tr></table><p>job description:</p><p>Build apps</p>"
        body = inbox.extract_body({"mimeType": "text/html", "body": {"data": encoded(html)}})
        result = inbox.parse_referral(body, [{"name": "from", "value": "=?utf-8?q?Recruiter_Name?= <jobs@example.com>"}])
        self.assertEqual(result["job_title"], "Developer")
        self.assertEqual(result["duration"], "6 months")
        self.assertEqual(result["sender"], "Recruiter Name <jobs@example.com>")
        self.assertEqual(result["jd_text"], "job description:\nBuild apps")

    def test_missing_fields_do_not_consume_next_label(self):
        result = inbox.parse_referral("Job Title ::\nDuration :: 6 months\nJob Description: Build systems", [])
        self.assertEqual(result["job_title"], "Unknown job title")
        self.assertEqual(result["sender"], "Unknown sender")
        self.assertEqual(result["jd_text"], "Job Description: Build systems")
        self.assertEqual(inbox.parse_referral("Unstructured email", [])["jd_text"], "Unstructured email")


class TwoStageWorkflowTests(unittest.IsolatedAsyncioTestCase):
    async def test_extract_returns_metadata_without_generation_then_tailor_uses_it(self):
        data = {"job_title": "Cloud Engineer", "sender": "Recruiter", "duration": "12 months",
                "jd_text": "Job Description: " + "Build reliable cloud infrastructure. " * 3}
        with patch.object(router, "_get_master_cv_or_400", return_value="master"), \
                patch.object(router, "fetch_latest_referral_jd", return_value=data) as fetch, \
                patch.object(router, "generate_tailored_resume", new=AsyncMock(return_value=b"%PDF")) as generate:
            extracted = await router.extract_latest_email(MagicMock())
            self.assertEqual(extracted.model_dump(), data)
            generate.assert_not_called()
            response = await router.tailor_extracted_email(extracted, MagicMock())
            self.assertEqual(response.body, b"%PDF")
            generate.assert_awaited_once_with(data["jd_text"], "master", job_title="Cloud Engineer")
            fetch.assert_called_once()

    async def test_short_isolated_jd_is_rejected_before_llm(self):
        data = inbox.parse_referral("Long introduction " * 100 + "\nJob Description: Short", [])
        with patch.object(router, "_get_master_cv_or_400"), \
                patch.object(router, "fetch_latest_referral_jd", return_value=data), \
                patch.object(router, "generate_tailored_resume", new=AsyncMock()) as generate:
            response = await router.extract_latest_email(MagicMock())
            self.assertEqual(response.status_code, 422)
            response = await router.tailor_extracted_email(router.Referral(**data), MagicMock())
            self.assertEqual(response.status_code, 422)
            generate.assert_not_called()
