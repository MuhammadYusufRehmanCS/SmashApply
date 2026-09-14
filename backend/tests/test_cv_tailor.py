import json
import unittest
from pathlib import Path
import sys
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import DEFAULT_OPENAI_MODEL, Settings
from app.models import MasterCV
from app.routers.jobs import _has_cached_tailoring, tailor_job
from app.services.cv_tailor import (
    LLMExecutionError,
    SYSTEM_PROMPT,
    TailoringError,
    TailorCVResult,
    _TailoredPayload,
    _prepare_summary_text,
    _reconstruct_tailored_text,
    _render_technical_expertise,
    _rewrite_experience_bullet,
    _split_experience_entries,
    _split_technical_expertise,
    _validate_tailored_payload,
    tailor_cv,
)
from app.services.text_sections import is_bullet_line


class CvTailorTests(unittest.TestCase):
    def test_blank_openai_model_falls_back_to_default(self):
        self.assertEqual(Settings(_env_file=None, openai_model="").openai_model, DEFAULT_OPENAI_MODEL)

    def test_replacement_character_bullets_are_editable(self):
        self.assertTrue(is_bullet_line("\ufffd Cloud & Infrastructure: AWS, Azure"))

        skills = _split_technical_expertise(
            "\ufffd Cloud & Infrastructure: AWS, Azure\n"
            "\ufffd DevOps & Platforms: Terraform, Docker"
        )
        self.assertIsNotNone(skills)
        self.assertEqual(skills[0]["label"], "Cloud & Infrastructure")
        self.assertEqual(skills[1]["items"], "Terraform, Docker")

        experience = _split_experience_entries(
            "Cloud Engineer | Arqon Consulting | Bay Area, CA | Jan 2025 - Present\n"
            "\ufffd Built CI/CD pipelines.\n"
            "with automated rollbacks.\n"
            "\ufffd Managed AWS infrastructure."
        )
        self.assertIsNotNone(experience)
        self.assertEqual(experience[0]["bullets"][0], "Built CI/CD pipelines. with automated rollbacks.")
        self.assertEqual(len(experience[0]["bullets"]), 2)

    def test_summary_cliche_starter_is_removed(self):
        summary = _prepare_summary_text(
            "Results-driven Cloud Engineer aligning AWS, Terraform, and CI/CD delivery."
        )
        self.assertEqual(summary, "Cloud Engineer aligning AWS, Terraform, and CI/CD delivery.")

    def test_system_prompt_requires_every_bullet_and_fixed_counts(self):
        self.assertIn("ALL Experience Bullets", SYSTEM_PROMPT)
        self.assertIn("Do not just swap 1-2 words", SYSTEM_PROMPT)
        self.assertIn("Never add, delete, merge, or split bullets/categories", SYSTEM_PROMPT)

    def test_system_prompt_requires_readable_keyword_use(self):
        self.assertIn("Readability takes priority over keyword density", SYSTEM_PROMPT)
        self.assertIn("DO NOT keyword-stuff", SYSTEM_PROMPT)

    def test_rewrite_does_not_stack_aligned_language_keywords(self):
        rewritten = _rewrite_experience_bullet(
            "Managed the lifecycle of containerized services using ACR/ECR and Nexus "
            "for secure artifact management.",
            [".NET", "Java", "C#", "React", "Node.js", "TypeScript", "ACR/ECR", "Nexus"],
        )
        plain = rewritten.replace("**", "")

        self.assertNotIn("-aligned", plain)
        self.assertNotRegex(plain, r"(?:\S+-aligned\s+){2,}")
        self.assertIn("containerized services", plain)
        self.assertIn("services for .NET and Java application delivery", plain)
        self.assertIn("ACR/ECR", plain)
        self.assertIn("Nexus", plain)

    def test_rewrite_keeps_cloud_database_keywords_in_readable_sentence(self):
        rewritten = _rewrite_experience_bullet(
            "Deployed Docker/Kubernetes workloads enabling 10-30s service readiness "
            "with scalable ingress routing.",
            ["AWS", "Azure", "Microsoft SQL Server", "Docker", "Kubernetes"],
        )
        plain = rewritten.replace("**", "")

        self.assertNotIn("-aligned", plain)
        self.assertNotIn("Docker/Kubernetes AWS Azure", plain)
        self.assertNotIn("SQL Server application delivery", plain)
        self.assertIn("service readiness", plain)
        self.assertIn("Docker/Kubernetes", plain)
        self.assertIn("across AWS and Azure infrastructure", plain)

    def test_devops_platforms_filters_cloud_service_duplicates(self):
        rendered = _render_technical_expertise(
            [
                {
                    "prefix": "-",
                    "label": "DevOps & Platforms",
                    "items": "Terraform, Helm, Docker, Kubernetes, CI/CD (GitHub Actions, Jenkins)",
                }
            ],
            ["AWS (Lambda, IAM, VPC), Terraform, Docker, GitHub Actions, CloudWatch"],
        )

        self.assertNotIn("AWS (Lambda", rendered)
        self.assertNotIn("CloudWatch", rendered)
        self.assertIn("Terraform", rendered)
        self.assertIn("GitHub Actions", rendered)

    def test_payload_validation_rejects_unchanged_bullet(self):
        payload = _TailoredPayload(
            keywords=["Terraform"],
            summary="Cloud automation engineer focused on Terraform delivery.",
            technical_expertise=["**Terraform**, Docker"],
            experience_bullets=[["Built CI/CD pipelines."]],
        )

        with self.assertRaises(TailoringError):
            _validate_tailored_payload(
                payload,
                summary_required=True,
                experience_entries=[{"bullets": ["Built CI/CD pipelines."]}],
                skills_entries=[
                    {"prefix": "-", "label": "DevOps & Platforms", "items": "Terraform, Docker"}
                ],
            )

    def test_payload_validation_rejects_unchanged_technical_expertise(self):
        payload = _TailoredPayload(
            keywords=["Terraform"],
            summary="Cloud automation engineer focused on Terraform delivery.",
            technical_expertise=["Terraform, Docker"],
            experience_bullets=[["Built **Terraform** CI/CD automation."]],
        )

        with self.assertRaises(TailoringError):
            _validate_tailored_payload(
                payload,
                summary_required=True,
                experience_entries=[{"bullets": ["Built CI/CD pipelines."]}],
                skills_entries=[
                    {"prefix": "-", "label": "DevOps & Platforms", "items": "Terraform, Docker"}
                ],
            )

    def test_payload_validation_rejects_bullets_without_target_keywords(self):
        payload = _TailoredPayload(
            keywords=["Terraform", "Docker", "CI/CD"],
            summary="Cloud automation engineer focused on Terraform-backed CI/CD delivery.",
            technical_expertise=["Docker, **Terraform**, **CI/CD**"],
            experience_bullets=[
                [
                    "Accelerated platform delivery through standardized release automation.",
                    "Strengthened infrastructure operations with repeatable delivery controls.",
                    "Improved deployment reliability through cleaner production workflows.",
                ]
            ],
        )

        with self.assertRaises(TailoringError):
            _validate_tailored_payload(
                payload,
                summary_required=True,
                experience_entries=[
                    {
                        "bullets": [
                            "Built release pipelines.",
                            "Managed infrastructure.",
                            "Improved releases.",
                        ]
                    }
                ],
                skills_entries=[
                    {"prefix": "-", "label": "DevOps & Platforms", "items": "Terraform, Docker, CI/CD"}
                ],
                target_keywords=["Terraform", "Docker", "CI/CD"],
            )

    def test_reconstruction_preserves_immutable_lines(self):
        sections = [
            {"name": "Header", "content": "MUHAMMAD YUSUF | CLOUD ENGINEER\nBay Area, CA"},
            {"name": "Executive Summary", "content": "Original summary."},
            {
                "name": "Technical Expertise",
                "content": "\ufffd Cloud & Infrastructure: AWS, Azure\n"
                "\ufffd DevOps & Platforms: Terraform, Docker",
            },
            {
                "name": "Professional Experience",
                "content": "Cloud Engineer | Arqon Consulting | Bay Area, CA | Jan 2025 - Present\n"
                "\ufffd Built CI/CD pipelines.",
            },
            {"name": "Education", "content": "\ufffd A.S. Computer Science, Los Angeles Harbor College"},
        ]
        payload = _TailoredPayload(
            keywords=["Terraform", "CI/CD"],
            summary="Cloud automation engineer focused on Terraform-backed CI/CD delivery.",
            technical_expertise=["Azure, **AWS**", "Docker, **Terraform**"],
            experience_bullets=[
                ["Accelerated **CI/CD** delivery by building **Terraform**-aligned pipelines."]
            ],
        )

        _validate_tailored_payload(
            payload,
            summary_required=True,
            experience_entries=_split_experience_entries(sections[3]["content"]),
            skills_entries=_split_technical_expertise(sections[2]["content"]),
        )
        tailored = _reconstruct_tailored_text(sections, payload)

        self.assertIn("MUHAMMAD YUSUF | CLOUD ENGINEER\nBay Area, CA", tailored)
        self.assertIn(
            "Cloud Engineer | Arqon Consulting | Bay Area, CA | Jan 2025 - Present",
            tailored,
        )
        self.assertIn("A.S. Computer Science, Los Angeles Harbor College", tailored)
        self.assertIn("Accelerated **CI/CD** delivery", tailored)

    def test_blank_keywords_cache_is_not_considered_tailored(self):
        job = type("JobStub", (), {"tailored_cv": "Original CV text", "tailored_keywords": ""})()
        self.assertFalse(_has_cached_tailoring(job))

        job.tailored_keywords = "Terraform, CI/CD"
        self.assertTrue(_has_cached_tailoring(job))


class CvTailorRetryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.sections = [
            {"name": "Header", "content": "MUHAMMAD YUSUF | CLOUD ENGINEER\nBay Area, CA"},
            {"name": "Executive Summary", "content": "Original summary."},
            {"name": "Professional Experience", "content":
             "Cloud Engineer | Arqon Consulting | Bay Area, CA | Jan 2025 - Present\n"
             "- Designed and governed high-availability AWS/Azure architectures, ensuring 99.9% uptime for production workloads.\n"
             "- Automated infrastructure with Terraform, Ansible, Python, and Bash to improve reliability and MTTR."},
            {"name": "Education", "content": "- A.S. Computer Science, Los Angeles Harbor College"},
        ]
        self.master = MasterCV(sections_json=json.dumps(self.sections), raw_text="", layout_json="{}")
        self.bullets = [
            "Sustained **99.9% uptime** by engineering resilient AWS/Azure systems for dependable production service delivery.",
            "Reduced recovery effort through repeatable provisioning and operational scripts built with Terraform, Ansible, Python, and Bash.",
        ]

    def payload(self):
        return _TailoredPayload(summary="Cloud engineer focused on reliable service operations.",
                                experience_bullets=[self.bullets.copy()])

    async def test_model_bullets_survive_without_stock_replacement(self):
        with patch("app.services.cv_tailor._request_tailored_payload", new_callable=AsyncMock) as request, \
             patch("app.services.cv_tailor._keywords_from_text", return_value=[]), \
             patch("app.services.cv_tailor._repair_tailored_payload") as repair:
            request.return_value = self.payload()
            result = await tailor_cv(self.master, "Systems Engineer", "Acme", "Reliable service operations")
        self.assertEqual(request.await_count, 1)
        repair.assert_not_called()
        for bullet in self.bullets:
            self.assertIn(bullet, result.text)
        self.assertTrue(result.cacheable)
        self.assertFalse(result.used_fallback)
        self.assertIn("Cloud Engineer | Arqon Consulting | Bay Area, CA | Jan 2025 - Present", result.text)
        self.assertIn("A.S. Computer Science, Los Angeles Harbor College", result.text)

    async def test_superficial_verb_swap_retries_with_feedback(self):
        weak = self.payload()
        weak.experience_bullets[0][0] = "Architected and governed high-availability AWS/Azure architectures, ensuring 99.9% uptime for production workloads."
        with patch("app.services.cv_tailor._request_tailored_payload", new_callable=AsyncMock) as request, \
             patch("app.services.cv_tailor._keywords_from_text", return_value=[]):
            request.side_effect = [weak, self.payload()]
            result = await tailor_cv(self.master, "Systems Engineer", "Acme", "Reliable service operations")
        self.assertEqual(request.await_count, 2)
        self.assertIn("previous response failed validation", request.await_args_list[1].args[1])
        self.assertIn(self.bullets[0], result.text)

    async def test_failed_retries_return_tailored_fallback_when_enabled(self):
        for failure in [TailoringError("invalid JSON"), LLMExecutionError("API unavailable")]:
            with self.subTest(failure=type(failure).__name__), \
                 patch("app.services.cv_tailor._request_tailored_payload", new_callable=AsyncMock) as request:
                request.side_effect = failure
                result = await tailor_cv(self.master, "Systems Engineer", "Acme", "Service operations", allow_fallback=True)
                self.assertEqual(request.await_count, 2)
                self.assertTrue(result.used_fallback)
                self.assertFalse(result.cacheable)
                self.assertNotIn("Original summary.", result.text)
                self.assertIn("Cloud Engineer | Arqon Consulting | Bay Area, CA | Jan 2025 - Present", result.text)

    async def test_tailor_route_requests_fallback_on_retry_failure(self):
        job = type("Job", (), {"id": 1, "title": "Systems Engineer", "company": "Acme", "description": "Service operations"})()
        fake_db = type("FakeDB", (), {"get": lambda self, model, job_id: job})()
        master = MasterCV(sections_json=json.dumps(self.sections), raw_text="", layout_json="{}")
        result = TailorCVResult(keywords=[], text="Tailored CV", cacheable=False, used_fallback=True)

        with patch("app.routers.jobs._get_master_cv_or_400", return_value=master), \
             patch("app.routers.jobs._run_tailor", new_callable=AsyncMock, return_value=result) as run_tailor:
            response = await tailor_job(1, db=fake_db)

        self.assertEqual(response.job_id, 1)
        self.assertEqual(response.tailored_cv, "Tailored CV")
        self.assertTrue(run_tailor.await_args.kwargs["allow_fallback"])

    async def test_missing_bullet_cannot_be_repaired_from_master(self):
        weak = self.payload()
        weak.experience_bullets[0].pop()
        with patch("app.services.cv_tailor._request_tailored_payload", new_callable=AsyncMock) as request, \
             patch("app.services.cv_tailor._keywords_from_text", return_value=[]):
            request.return_value = weak
            with self.assertRaises(TailoringError):
                await tailor_cv(self.master, "Systems Engineer", "Acme", "Service operations")
        self.assertEqual(request.await_count, 2)

    def test_old_cached_stock_bullets_are_not_reused(self):
        text = _reconstruct_tailored_text(self.sections, _TailoredPayload())
        job = type("JobStub", (), {"tailored_cv": text, "tailored_keywords": "AWS"})()
        self.assertFalse(_has_cached_tailoring(job, self.master))
        job.tailored_cv = _reconstruct_tailored_text(self.sections, self.payload())
        self.assertTrue(_has_cached_tailoring(job, self.master))


if __name__ == "__main__":
    unittest.main()
