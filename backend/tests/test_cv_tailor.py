import json
import unittest
from pathlib import Path
import sys
from unittest.mock import AsyncMock, patch
from types import SimpleNamespace

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
    _clean_field_text,
    _enforce_category_contract,
    _keyword_fits_category,
    _is_valid_category_rewording,
    _build_prompt,
    _reconstruct_tailored_text,
    _render_technical_expertise,
    _render_experience_entries,
    _rewrite_experience_bullet,
    _split_experience_entries,
    _split_technical_expertise,
    _validate_tailored_payload,
    _request_tailored_payload,
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
        self.assertIn("Rewrite every bullet", SYSTEM_PROMPT)
        self.assertIn("Arqon Consulting = EXACTLY 4 bullets", SYSTEM_PROMPT)
        self.assertIn("Ventera Group = EXACTLY 3 bullets", SYSTEM_PROMPT)

    def test_system_prompt_requires_jd_keywords_action_verbs_and_no_durations(self):
        self.assertIn("Reuse that exact JD wording", SYSTEM_PROMPT)
        self.assertIn("strong past-tense action verb", SYSTEM_PROMPT)
        self.assertIn("Never mention years of experience", SYSTEM_PROMPT)
        self.assertIn("no HTML tags", SYSTEM_PROMPT)
        self.assertIn("mention each specific keyword EXACTLY ONCE", SYSTEM_PROMPT)

    def test_prompt_and_schema_allow_one_tool_mention_in_skills_or_experience(self):
        from app.services.cv_schema import finalized_schema
        self.assertTrue(SYSTEM_PROMPT.startswith("STRICT HARD CONSTRAINT (ONE MENTION PER TOOL)"))
        self.assertIn("Maximum 1 total mention per tool across all fields", SYSTEM_PROMPT)
        self.assertIn("STRATEGIC DISTRIBUTION", SYSTEM_PROMPT)
        props = finalized_schema()["properties"]
        experience = props["experience_bullets"]["properties"]
        for prop in (props["core_skills"]["items"], experience["arqon"]["items"], experience["ventera"]["items"]):
            self.assertIn("at most once in the whole document", prop["description"])
        for prop in (props["summary"], props["role_title"]):
            self.assertIn("Name no specific tools", prop["description"])

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

    def test_skills_rendering_preserves_model_tools(self):
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

        self.assertIn("AWS (Lambda", rendered)
        self.assertIn("CloudWatch", rendered)
        self.assertIn("Terraform", rendered)
        self.assertIn("GitHub Actions", rendered)

    def test_payload_validation_accepts_unchanged_bullet(self):
        payload = _TailoredPayload(
            keywords=["Terraform"],
            summary="Cloud automation engineer focused on Terraform delivery.",
            technical_expertise=["**Terraform**, Docker", "Delivery engineering", "Leadership & Cross-Functional Collaboration"],
            experience_bullets=[["Built CI/CD pipelines."]],
        )

        _validate_tailored_payload(
            payload,
            summary_required=True,
            experience_entries=[{"bullets": ["Built CI/CD pipelines."]}],
            skills_entries=[
                {"prefix": "-", "label": "DevOps & Platforms", "items": "Terraform, Docker"}
            ],
        )

    def test_payload_validation_accepts_unchanged_technical_expertise(self):
        payload = _TailoredPayload(
            keywords=["Terraform"],
            summary="Cloud automation engineer focused on Terraform delivery.",
            technical_expertise=["Terraform, Docker", "Delivery engineering", "Leadership & Cross-Functional Collaboration"],
            experience_bullets=[["Built **Terraform** CI/CD automation."]],
        )

        _validate_tailored_payload(
            payload,
            summary_required=True,
            experience_entries=[{"bullets": ["Built CI/CD pipelines."]}],
            skills_entries=[
                {"prefix": "-", "label": "DevOps & Platforms", "items": "Terraform, Docker"}
            ],
        )

    def test_payload_validation_accepts_bullets_without_target_keywords(self):
        payload = _TailoredPayload(
            keywords=["Terraform", "Docker", "CI/CD"],
            summary="Cloud automation engineer focused on Terraform-backed CI/CD delivery.",
            technical_expertise=["Docker, **Terraform**, **CI/CD**", "Cloud architecture", "Leadership & Cross-Functional Collaboration"],
            experience_bullets=[
                [
                    "Accelerated platform delivery through standardized release automation.",
                    "Strengthened infrastructure operations with repeatable delivery controls.",
                    "Improved deployment reliability through cleaner production workflows.",
                ]
            ],
        )

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
            technical_expertise=["Azure, **AWS**", "Docker, **Terraform**", "Leadership & Cross-Functional Collaboration"],
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

    def test_nonempty_reworded_bullets_survive_count_mismatch(self):
        entries = [{"header_line": "Engineer | Company | 2025", "tagline": None,
                    "bullets": ["Original first.", "Original second."]}]
        rendered = _render_experience_entries(entries, [["Model-written replacement."]])
        self.assertIn("Model-written replacement.", rendered)
        self.assertNotIn("Original", rendered)
        with self.assertLogs("app.services.cv_tailor", level="ERROR") as logs:
            with self.assertRaisesRegex(TailoringError, "employer 1"):
                _render_experience_entries(entries, [[]])
        self.assertIn("missing or empty model bullets", logs.output[0])

    def test_reconstruction_preserves_summary_and_short_skills_verbatim(self):
        sections = [
            {"name": "Executive Summary", "content": "Original summary."},
            {"name": "Technical Expertise", "content": "- Cloud: AWS, Azure, GCP, networking, monitoring, architecture"},
        ]
        payload = _TailoredPayload(summary="Results-driven engineer with specific model wording.",
                                  technical_expertise=["**GCP**", "Delivery engineering", "Leadership & Cross-Functional Collaboration"])
        before = payload.model_dump()
        _validate_tailored_payload(payload, True, None, _split_technical_expertise(sections[1]['content']))
        self.assertEqual(payload.model_dump(), before)
        rendered = _reconstruct_tailored_text(sections, payload)
        self.assertIn(payload.summary, rendered)
        self.assertIn("Cloud: **GCP**", rendered)
        self.assertNotIn("AWS", rendered)

    def test_long_and_alternate_headers_include_verified_source_facts(self):
        headers = [
            "Senior Cloud Infrastructure Engineer | Very Long Employer Name Consulting and Technology Services | San Francisco Bay Area, California | January 2023 - September 2025",
            "Cloud Engineer at Example Consulting",
            "Cloud Engineer - Example Consulting - Jan 2023 to Sep 2025",
            "Cloud Engineer|Example Consulting",
            "Cloud Engineer\nExample Consulting\nJan 2023 \u2013 Sep 2025",
        ]
        for header in headers:
            with self.subTest(header=header):
                content = header + "\n- Delivered reliable services.\n- Automated deployments."
                entries = _split_experience_entries(content)
                self.assertEqual(len(entries), 1)
                self.assertEqual(entries[0]['bullets'], ['Delivered reliable services.', 'Automated deployments.'])
                prompt, parsed, _ = _build_prompt([{'name':'Professional Experience','content':content}],
                                                  'Cloud Engineer', 'Target', 'Cloud infrastructure')
                self.assertIn('Delivered reliable services.', prompt)
                self.assertIn('Automated deployments.', prompt)
                self.assertIn('"bullet_count_required": 2', prompt)
                self.assertEqual(len(parsed[0]['bullets']), 2)

    def test_multiline_employer_boundaries_do_not_swallow_bullets(self):
        text = ("Cloud Engineer\nExample A\nJan 2023 - Feb 2024\n- Delivered services\nwith automated recovery.\n"
                "Platform Engineer\nExample B\nMar 2024 - Present\n- Built release tools.")
        entries = _split_experience_entries(text)
        self.assertEqual(len(entries), 2)
        self.assertIn('Example B', entries[1]['header_line'])
        self.assertEqual(entries[0]['bullets'], ['Delivered services with automated recovery.'])

    def test_unparseable_experience_fails_visibly_before_model_request(self):
        with self.assertLogs('app.services.cv_tailor', level='ERROR'):
            with self.assertRaisesRegex(TailoringError, 'Could not parse'):
                _build_prompt([{'name':'Professional Experience','content':'Unstructured text without headers or bullets'}],
                              'Engineer', 'Target', 'Job description')

    def test_uppercase_wrapped_headers_stay_inside_experience_section(self):
        from app.services.text_sections import segment_sections
        sections = segment_sections('PROFESSIONAL EXPERIENCE\nCLOUD ENGINEER\nEXAMPLE CONSULTING\nJan 2023 - Present\n- Built systems.\nADDITIONAL\nEnglish')
        self.assertEqual([section['name'] for section in sections], ['Professional Experience', 'Additional'])
        entries = _split_experience_entries(sections[0]['content'])
        self.assertIn('EXAMPLE CONSULTING', entries[0]['header_line'])
        self.assertEqual(entries[0]['bullets'], ['Built systems.'])

    def test_content_cleaning_preserves_reframed_phrasing(self):
        text = 'Reframed infrastructure operations (rewords existing bullet) with 8 years of experience in new technologies.'
        self.assertEqual(_clean_field_text(text), text)
        self.assertEqual(_clean_field_text('```text\n' + text + '\n```'), text)
        self.assertTrue(_is_valid_category_rewording('Rust', {'label':'Languages', 'items':'A very long original technology list'}))

    def test_category_contract_preserves_domain_terms_without_blacklists(self):
        candidates = [
            'Data Platform, Orchestration, Lineage, AWS (Glue, Athena), Terraform',
            '**Kafka**, Kubernetes, Data Pipeline, Governance, AWS, Lineage',
            '  Lineage, Lineage, New Domain Tool (alpha, beta)  ',
            '',
        ]
        for label in ['Cloud & Infrastructure', 'DevOps & Platforms', 'Monitoring & Security', 'Languages & Tools']:
            for candidate in candidates:
                with self.subTest(label=label, candidate=candidate):
                    self.assertEqual(_enforce_category_contract(label, candidate, 'ORIGINAL TOOLS'), candidate)

    def test_domain_terms_are_not_blocked_by_category_membership(self):
        for label in ['Cloud & Infrastructure', 'DevOps & Platforms', 'Monitoring & Security', 'Languages & Tools']:
            for term in ['Data Platform', 'Orchestration', 'Lineage', 'Terraform', 'AWS', 'Novel Domain Term']:
                with self.subTest(label=label, term=term):
                    self.assertTrue(_keyword_fits_category(label, term))
        self.assertFalse(_keyword_fits_category('Cloud', '   '))

    def test_invalid_experience_output_logs_error_without_original_substitution(self):
        entries = [{'header_line':'Engineer | Company | 2025', 'tagline':None,
                    'bullets':['ORIGINAL BULLET']}]
        for payload in [None, [], [[]], [['']], [['New wording', ' ']], [[None]], ['not a bullet list'], [['One'], ['Extra employer']]]:
            with self.subTest(payload=payload), self.assertLogs('app.services.cv_tailor', level='ERROR') as logs:
                with self.assertRaises(TailoringError):
                    _render_experience_entries(entries, payload)
            self.assertIn('no original bullets substituted', logs.output[0])

    def test_blank_keywords_cache_is_not_considered_tailored(self):
        job = type("JobStub", (), {"tailored_cv": "Original CV text", "tailored_keywords": ""})()
        self.assertFalse(_has_cached_tailoring(job))

        job.tailored_keywords = "Terraform, CI/CD"
        self.assertTrue(_has_cached_tailoring(job))


class CvTailorRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_full_jd_and_verified_experience_reach_api(self):
        jd = 'Affirm data platform requirements\n' + 'Streaming ingestion and schema governance.\n' * 300 + 'FINAL REQUIREMENT: data lineage'
        payload = self.payload()
        payload.experience_bullets = [[
            'Architected **Kafka** ingestion with schema validation and lineage tracking to prevent corrupt data from reaching analytics.',
            'Built **Airflow** recovery workflows with replayable checkpoints and data-quality gates to restore interrupted pipelines.',
        ]]
        client = AsyncMock()
        client.chat.completions.create.return_value = SimpleNamespace(choices=[
            SimpleNamespace(message=SimpleNamespace(content=payload.model_dump_json()))])
        with patch('app.services.cv_tailor.AsyncOpenAI') as factory, \
             patch('app.services.cv_tailor.get_settings', return_value=SimpleNamespace(openai_api_key='test', openai_model='gpt-4o')), \
             patch('app.services.cv_tailor._repair_tailored_payload', side_effect=AssertionError('No deterministic overwrite')), \
             patch('app.services.cv_tailor._is_superficial_rewrite', side_effect=AssertionError('No similarity filter')):
            factory.return_value.__aenter__.return_value = client
            result = await tailor_cv(self.master, 'Data Engineer', 'Affirm', jd)
        request = client.chat.completions.create.await_args.kwargs
        user = next(message['content'] for message in request['messages'] if message['role'] == 'user')
        data = json.loads(user)
        self.assertEqual(data['job_description'], jd)
        self.assertEqual(data['experience_requirements'][0]['bullet_count_required'], 4)
        for entry in _split_experience_entries(self.sections[2]['content']):
            for bullet in entry['bullets']:
                self.assertIn(bullet, data['experience_requirements'][0]['verified_bullets'])
        self.assertEqual(request['temperature'], 0.1)
        self.assertTrue(request['response_format']['json_schema']['strict'])
        self.assertEqual(result.template_data['experience_bullets'], payload.experience_bullets)

    async def test_api_receives_category_context_and_requested_temperature(self):
        sections = [{'name': 'Technical Expertise', 'content':
                     '- Cloud & Infrastructure: AWS, Azure\nGCP, VPC\n- DevOps & Platforms: Terraform, Kubernetes'}]
        prompt, _, entries = _build_prompt(sections, 'Data Engineer', 'Example', 'Data platforms')
        self.assertEqual(entries[0]['items'], 'AWS, Azure GCP, VPC')
        client = AsyncMock()
        client.chat.completions.create.return_value = SimpleNamespace(choices=[
            SimpleNamespace(message=SimpleNamespace(content=json.dumps({
                'technical_expertise': ['AWS, BigQuery', 'Airflow, Terraform', 'Leadership & Cross-Functional Collaboration']})))])
        with patch('app.services.cv_tailor.AsyncOpenAI') as factory:
            factory.return_value.__aenter__.return_value = client
            await _request_tailored_payload(SimpleNamespace(openai_api_key='test', openai_model='gpt-4o'), prompt)
        args = client.chat.completions.create.await_args.kwargs
        self.assertEqual(args['temperature'], 0.1)
        self.assertEqual(json.loads(args["messages"][1]["content"])["verified_master_cv"], sections)

    def test_long_new_workstreams_survive_rendering_without_content_checks(self):
        entries = [{'header_line': 'Engineer | Example | 2023 - Present', 'tagline': None,
                    'bullets': ['Original bullet.']}]
        rewritten = ('Architected **Data Platform** orchestration with Airflow, lineage tracking, '
                     'schema contracts, and policy-as-code checks to ensure auditability. ' * 4).strip()
        with patch('app.services.cv_tailor._is_superficial_rewrite', side_effect=AssertionError('No similarity checks')), \
             patch('app.services.cv_tailor._clean_field_text', side_effect=AssertionError('No regex content cleaning')):
            rendered = _render_experience_entries(entries, [[rewritten]])
        self.assertIn(rewritten, rendered)
        self.assertNotIn('Original bullet.', rendered)

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
                                technical_expertise=["Cloud architecture", "Delivery engineering", "Leadership & Cross-Functional Collaboration"],
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

    async def test_model_wording_is_not_rejected_for_similarity(self):
        weak = self.payload()
        weak.experience_bullets[0][0] = "Architected and governed high-availability AWS/Azure architectures, ensuring 99.9% uptime for production workloads."
        with patch("app.services.cv_tailor._request_tailored_payload", new_callable=AsyncMock) as request, \
             patch("app.services.cv_tailor._keywords_from_text", return_value=[]):
            request.side_effect = [weak, self.payload()]
            result = await tailor_cv(self.master, "Systems Engineer", "Acme", "Reliable service operations")
        self.assertEqual(request.await_count, 1)
        self.assertIn(weak.experience_bullets[0][0], result.text)

    async def test_api_failure_raises_after_one_call(self):
        with patch("app.services.cv_tailor._request_tailored_payload", new_callable=AsyncMock) as request:
            request.side_effect = LLMExecutionError("API unavailable")
            with self.assertRaises(LLMExecutionError):
                await tailor_cv(self.master, "Systems Engineer", "Acme", "Service operations", allow_fallback=True)
        self.assertEqual(request.await_count, 1)

    async def test_invalid_json_falls_back_to_master_without_retry(self):
        for content in ("not json", '{"summary": "Only a summary."}', "[]", ""):
            client = AsyncMock()
            client.chat.completions.create.return_value = SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
            with self.subTest(content=content), patch("app.services.cv_tailor.AsyncOpenAI") as factory, \
                 patch("app.services.cv_tailor.get_settings",
                       return_value=SimpleNamespace(openai_api_key="test", openai_model="gpt-4o")):
                factory.return_value.__aenter__.return_value = client
                result = await tailor_cv(self.master, "Systems Engineer", "Acme", "Service operations")
                self.assertEqual(client.chat.completions.create.await_count, 1)
                self.assertTrue(result.used_fallback)
                self.assertFalse(result.cacheable)
                for bullet in _split_experience_entries(self.sections[2]["content"])[0]["bullets"]:
                    self.assertIn(bullet, result.text)

    async def test_missing_key_uses_master_value_and_strips_html_and_durations(self):
        data = self.payload().model_dump(exclude={"experience_bullets"})
        data["summary"] = "<b>Cloud engineer</b> with 5+ years of experience in reliable operations."
        client = AsyncMock()
        client.chat.completions.create.return_value = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(data)))])
        master = _TailoredPayload.model_construct(role_title="", keywords=[], summary="Master summary.",
                                                  core_skills=[], experience_bullets=[["Master bullet."]])
        with patch("app.services.cv_tailor.AsyncOpenAI") as factory:
            factory.return_value.__aenter__.return_value = client
            payload = await _request_tailored_payload(
                SimpleNamespace(openai_api_key="test", openai_model="gpt-4o"), "Prompt", fallback=master)
        self.assertEqual(client.chat.completions.create.await_count, 1)
        self.assertEqual(payload.experience_bullets, [["Master bullet."]])
        self.assertNotIn("<b>", payload.summary)
        self.assertNotIn("years", payload.summary)
        self.assertIn("Cloud engineer", payload.summary)

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
            result = await tailor_cv(self.master, "Systems Engineer", "Acme", "Service operations")
        self.assertEqual(request.await_count, 1)
        self.assertTrue(result.used_fallback)
        self.assertFalse(result.cacheable)

    def test_cache_validation_checks_structure_not_wording(self):
        text = _reconstruct_tailored_text(self.sections, self.payload())
        job = type("JobStub", (), {"tailored_cv": text, "tailored_keywords": "AWS"})()
        self.assertTrue(_has_cached_tailoring(job, self.master))
        job.tailored_cv = _reconstruct_tailored_text(self.sections, self.payload())
        self.assertTrue(_has_cached_tailoring(job, self.master))


if __name__ == "__main__":
    unittest.main()
