import json
import unittest
from pathlib import Path
import sys
from unittest.mock import AsyncMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import DEFAULT_OPENAI_MODEL, Settings
from app.models import MasterCV
from app.routers.jobs import _has_cached_tailoring
from app.services.cv_tailor import (
    LLMExecutionError,
    SYSTEM_PROMPT,
    TailoringError,
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

    def test_system_prompt_forbids_summary_only_tailoring(self):
        self.assertIn("Modifying ONLY the Executive Summary is an automatic failure", SYSTEM_PROMPT)
        self.assertIn(
            "Rewrite the content using strong action verbs, target keywords, and tailored phrasing.",
            SYSTEM_PROMPT,
        )

    def test_system_prompt_requires_recruiter_readable_truthful_tailoring(self):
        self.assertIn("ATS-aware, recruiter-readable", SYSTEM_PROMPT)
        self.assertIn("Do not keyword-stuff", SYSTEM_PROMPT)
        self.assertIn("Do not invent experience to satisfy the posting", SYSTEM_PROMPT)
        self.assertIn("emphasize the closest truthful adjacent experience", SYSTEM_PROMPT)

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
    async def test_tailor_cv_retries_once_then_uses_deterministic_tailoring(self):
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
        master_cv = MasterCV(sections_json=json.dumps(sections), raw_text="", layout_json="{}")

        with patch("app.services.cv_tailor._request_tailored_payload", new_callable=AsyncMock) as request:
            request.side_effect = [
                TailoringError("malformed json"),
                TailoringError("strict bullet count validation failed"),
            ]

            result = await tailor_cv(
                master_cv,
                "Cloud Engineer",
                "Acme",
                "Need Terraform, Docker, CI/CD, and AWS.",
                allow_fallback=True,
            )

        self.assertEqual(result.keywords, ["Terraform", "Docker", "CI/CD", "AWS"])
        self.assertTrue(result.cacheable)
        self.assertFalse(result.used_fallback)
        self.assertEqual(request.await_count, 2)
        self.assertEqual([call.args[2] for call in request.await_args_list], [0.3, 0.3])
        self.assertIn(
            "Cloud Engineer focused on cloud infrastructure, automation, and production delivery "
            "using Terraform, Docker, CI/CD, AWS",
            result.text,
        )
        self.assertIn("MUHAMMAD YUSUF | CLOUD ENGINEER", result.text)
        self.assertIn(
            "Cloud Engineer | Arqon Consulting | Bay Area, CA | Jan 2025 - Present",
            result.text,
        )
        self.assertIn("A.S. Computer Science, Los Angeles Harbor College", result.text)
        self.assertIn("Engineered **CI/CD** pipelines.", result.text)

    async def test_retry_repairs_unchanged_skills_and_bullets(self):
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
        ]
        master_cv = MasterCV(sections_json=json.dumps(sections), raw_text="", layout_json="{}")
        retry_payload = _TailoredPayload(
            keywords=["Terraform", "CI/CD", "AWS"],
            summary="Cloud automation engineer focused on AWS-backed Terraform and CI/CD delivery.",
            technical_expertise=["**AWS**, Azure", "**Terraform**, Docker"],
            experience_bullets=[["Built CI/CD pipelines."]],
        )

        with patch("app.services.cv_tailor._request_tailored_payload", new_callable=AsyncMock) as request:
            request.side_effect = [
                TailoringError("malformed json"),
                retry_payload,
            ]

            result = await tailor_cv(
                master_cv,
                "Cloud Engineer",
                "Acme",
                "Need Terraform, Docker, CI/CD, and AWS.",
            )

        self.assertEqual(request.await_count, 2)
        self.assertTrue(result.cacheable)
        self.assertFalse(result.used_fallback)
        self.assertIn("Cloud automation engineer", result.text)
        self.assertIn("**Terraform**, **Docker**, **CI/CD**", result.text)
        self.assertIn("Engineered **CI/CD** pipelines.", result.text)

    async def test_parseable_payload_is_repaired_before_validation(self):
        sections = [
            {"name": "Header", "content": "MUHAMMAD YUSUF | CLOUD ENGINEER\nBay Area, CA"},
            {"name": "Executive Summary", "content": "Original summary."},
            {
                "name": "Technical Expertise",
                "content": "\ufffd Cloud & Infrastructure: AWS, Azure\n"
                "\ufffd DevOps & Platforms: Terraform, Docker, Kubernetes, CI/CD, GitHub Actions, Jenkins",
            },
            {
                "name": "Professional Experience",
                "content": "Cloud Engineer | Arqon Consulting | Bay Area, CA | Jan 2025 - Present\n"
                "\ufffd Designed and governed high-availability AWS/Azure architectures.\n"
                "\ufffd Optimized GitHub Actions/Jenkins pipelines with SonarQube quality gates.\n"
                "\ufffd Built automated artifact management and release pipelines.\n"
                "\ufffd Automated infrastructure with Terraform, Ansible, Python, and Bash.",
            },
        ]
        master_cv = MasterCV(sections_json=json.dumps(sections), raw_text="", layout_json="{}")
        weak_payload = _TailoredPayload(
            keywords=[],
            summary="Cloud automation engineer focused on production delivery.",
            technical_expertise=["AWS, Azure", "Terraform, Docker, Kubernetes, CI/CD, GitHub Actions, Jenkins"],
            experience_bullets=[
                [
                    "Designed and governed high-availability AWS/Azure architectures.",
                    "Optimized GitHub Actions/Jenkins pipelines with SonarQube quality gates.",
                    "Built automated artifact management and release pipelines.",
                    "Automated infrastructure with Terraform, Ansible, Python, and Bash.",
                ]
            ],
        )

        with patch("app.services.cv_tailor._request_tailored_payload", new_callable=AsyncMock) as request:
            request.return_value = weak_payload

            result = await tailor_cv(
                master_cv,
                "Cloud Engineer",
                "Acme",
                "Need AWS, Terraform, Docker, Kubernetes, CI/CD, release automation, and SonarQube.",
            )

        self.assertEqual(request.await_count, 1)
        self.assertTrue(result.cacheable)
        self.assertFalse(result.used_fallback)
        self.assertIn("Architected and governed", result.text)
        self.assertIn("**CI/CD** pipelines", result.text)
        self.assertIn("Codified and automated infrastructure with **Terraform**", result.text)

    async def test_tailor_cv_without_fallback_uses_deterministic_tailoring_when_retry_has_no_json(self):
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
        ]
        master_cv = MasterCV(sections_json=json.dumps(sections), raw_text="", layout_json="{}")

        with patch("app.services.cv_tailor._request_tailored_payload", new_callable=AsyncMock) as request:
            request.side_effect = [
                TailoringError("malformed json"),
                TailoringError("malformed retry"),
            ]

            result = await tailor_cv(
                master_cv,
                "Cloud Engineer",
                "Acme",
                "Need Terraform, Docker, CI/CD, and AWS.",
            )

        self.assertEqual(request.await_count, 2)
        self.assertTrue(result.cacheable)
        self.assertFalse(result.used_fallback)
        self.assertIn(
            "Cloud Engineer focused on cloud infrastructure, automation, and production delivery "
            "using Terraform, Docker, CI/CD, AWS",
            result.text,
        )
        self.assertIn("Engineered **CI/CD** pipelines.", result.text)

    async def test_openai_execution_error_falls_back_without_raw_failure(self):
        sections = [
            {"name": "Header", "content": "MUHAMMAD YUSUF | CLOUD ENGINEER\nBay Area, CA"},
            {"name": "Executive Summary", "content": "Original summary."},
            {
                "name": "Professional Experience",
                "content": "Cloud Engineer | Arqon Consulting | Bay Area, CA | Jan 2025 - Present\n"
                "\ufffd Built CI/CD pipelines.",
            },
        ]
        master_cv = MasterCV(sections_json=json.dumps(sections), raw_text="", layout_json="{}")

        with patch("app.services.cv_tailor._request_tailored_payload", new_callable=AsyncMock) as request:
            request.side_effect = [
                LLMExecutionError("OPENAI_API_KEY is missing; set it in backend/.env."),
                LLMExecutionError("OPENAI_API_KEY is missing; set it in backend/.env."),
            ]

            result = await tailor_cv(
                master_cv,
                "Cloud Engineer",
                "Acme",
                "Need Terraform, Docker, CI/CD, and AWS.",
            )

        self.assertEqual(request.await_count, 2)
        self.assertFalse(result.cacheable)
        self.assertTrue(result.used_fallback)
        self.assertIn("Original summary.", result.text)


if __name__ == "__main__":
    unittest.main()
