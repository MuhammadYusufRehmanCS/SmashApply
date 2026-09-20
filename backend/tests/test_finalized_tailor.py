import io
import json
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pdfplumber
from pypdf import PdfReader

from app.services import tailor
from app.services.cv_tailor import _TailoredPayload, TailoringError, LLMExecutionError
from app.services.cv_tailor import compute_match_score, jd_keywords, template_context_from_text, tailor_cv
from app.services.pdf_generator import build_ats_pdf, render_cv_html, CVOverflowError


def candidate():
    return _TailoredPayload(
        role_title="Cloud Engineer", keywords=["Jenkins", "Terraform"],
        summary="Cloud engineer delivering reliable infrastructure and efficient operations, aligning technical execution with business needs through practical automation expertise and accountable service ownership.",
        technical_expertise=[
            "Cloud Architecture: Resilient service design and workload capacity planning.",
            "Delivery Engineering: Release governance and production readiness practices.",
            "Leadership & Cross-Functional Collaboration: Coordinate engineering teams and communicate operational priorities.",
        ], experience_bullets=[[
            "Designed high-availability architectures supporting 99.9% uptime for production workloads.",
            "Optimized GitHub Actions workflows with SonarQube quality gates for deployments under 60 seconds.",
            "Automated maintenance with Ansible, Python and Bash to reduce recovery time.",
            "Selected Project: Release Automation System - Built Jenkins pipelines ensuring complete artifact traceability.",
        ], [
            "Improved release delivery efficiency by 60% while reducing enterprise deployment errors.",
            "Enabled container service readiness in 10-30 seconds through scalable ingress routing.",
            "Selected Project: Automated Infrastructure Provisioning - Built modular Terraform scripts enabling environment setup in 5-10 minutes.",
        ]],
    )


def master():
    payload = candidate()
    sections = [
        {"name": "Header", "content": "TEST CANDIDATE | CLOUD ENGINEER | TERRAFORM\nBay Area | test@example.com\n1/1"},
        {"name": "Professional Experience", "content":
         "Arqon Consulting | DevOps Engineer | Jan 2025 - Present\n" +
         "\n".join("- " + b for b in payload.experience_bullets[0]) +
         "\nVentera Group | Cloud Engineer | Aug 2023 - Jan 2025\n" +
         "\n".join("- " + b for b in payload.experience_bullets[1])},
        {"name": "Education & Certifications", "content": "- A.S. Computer Science & Engineering, Example College, 2023\n- Certifications: AWS Certified Cloud Practitioner (CCP)\n- Tooling: Docker, Kubernetes"},
        {"name": "Additional Information", "content": "- Languages: English (Fluent)\n- Core Strengths: remove this"},
    ]
    return SimpleNamespace(sections_json=json.dumps(sections), raw_text=json.dumps(sections))


class FinalizedValidationTests(unittest.TestCase):
    def test_python_shortens_title_and_renderer_rejects_broken_structure(self):
        payload = candidate()
        payload.role_title = 'Senior Cloud Platform Infrastructure Operations Engineer'
        context = tailor._context(tailor._source_context(master()), payload)
        self.assertLessEqual(len(context['role_title'].split()), 4)
        html = render_cv_html(context)
        self.assertIn('MUHAMMAD YUSUF', html)
        self.assertIn('MULTI-CLOUD | TERRAFORM | CI/CD | CLOUD AUTOMATION', html)
        self.assertIn('A.S Computer Science &amp; Engineering | Bay Area, CA', html)
        broken = deepcopy(context)
        broken['core_skills'].pop()
        with self.assertRaisesRegex(TailoringError, 'three Core Skills'):
            render_cv_html(broken)

    def test_permitted_emphasis_and_project_dashes_do_not_reject_payload(self):
        payload = candidate()
        payload.core_skills[2] = payload.core_skills[2].replace(
            'Leadership & Cross-Functional Collaboration:',
            '**Leadership & Cross-Functional Collaboration:**')
        for group in payload.experience_bullets:
            group[-1] = group[-1].replace('Selected Project:', '**Selected Project:**').replace(' - ', ' \u2013 ')
        context = tailor._context(tailor._source_context(master()), payload, master().raw_text)
        self.assertTrue(context['core_skills'][2].startswith(tailor.LEADERSHIP))
        for entry, prefix in zip(context['experience'], tailor.PROJECTS):
            self.assertTrue(entry['bullets'][-1].startswith(prefix))
        self.assertEqual(len(PdfReader(io.BytesIO(build_ats_pdf(context))).pages), 1)

    def test_retry_feedback_lists_all_repeated_tools(self):
        payload = candidate()
        payload.core_skills[0] += ' Terraform, Jenkins, Ansible.'
        with self.assertRaises(tailor.FinalizedValidationError) as caught:
            tailor._validate(payload)
        for tool in ('Terraform', 'Jenkins', 'Ansible'):
            self.assertIn(tool, str(caught.exception))

    def test_failure_detail_identifies_local_validation_reason(self):
        from app.routers.jobs import _tailoring_failure_detail
        error = TailoringError('Finalized CV did not pass structure validation after three attempts.')
        error.__cause__ = tailor.FinalizedValidationError('Repeated technology; keep it in its Selected Project when applicable: Terraform')
        self.assertIn('Repeated technology', _tailoring_failure_detail(error))
        self.assertNotIn('after three attempts', _tailoring_failure_detail(error))

    def test_new_core_skills_key_and_legacy_alias_share_one_schema(self):
        payload = candidate()
        data = payload.model_dump()
        self.assertIn('core_skills', data)
        self.assertNotIn('technical_expertise', data)
        self.assertEqual(_TailoredPayload.model_validate(data).core_skills, payload.core_skills)

    def test_title_budgets_and_unsupported_claims_are_rejected(self):
        for edit in (
            lambda p: setattr(p, 'role_title', 'Senior Cloud Platform Infrastructure Security Operations Engineering Manager'),
            lambda p: p.experience_bullets[0].__setitem__(0, 'Improved availability to 99.999% through resilient service design.'),
            lambda p: p.experience_bullets[0].__setitem__(0, 'Implemented NIST compliance controls across production workloads.'),
            lambda p: p.experience_bullets[0].__setitem__(0, 'Built Snowflake pipelines for analytics workloads.'),
            lambda p: p.core_skills.__setitem__(0, 'Cloud Architecture: ' + 'service ' * 29),
        ):
            payload = candidate()
            edit(payload)
            with self.assertRaises(TailoringError):
                tailor._validate(payload, master().raw_text)

    def test_saved_text_keeps_finalized_structure_for_job_download(self):
        context = tailor._context(tailor._source_context(master()), candidate())
        restored = template_context_from_text(tailor._serialize(context))
        self.assertTrue(restored['finalized'])
        self.assertEqual(restored['core_skills'], context['core_skills'])
        self.assertEqual([len(e['bullets']) for e in restored['experience']], [4, 3])
        self.assertEqual(restored['education'], context['education'])

    def test_keyword_score_counts_unique_aliases_and_unmet_requirements(self):
        keywords = jd_keywords('AWS, Kubernetes and SOC 2 security controls',
                               ['Amazon Web Services', 'AWS', 'Kubernetes', 'SOC 2', 'security controls', 'invented term'])
        self.assertNotIn('invented term', keywords)
        self.assertEqual(len(keywords), 4)
        self.assertEqual(compute_match_score(keywords, '**AWS** and K8s security controls'), 75)
        self.assertEqual(compute_match_score(keywords, 'AWS AWS AWS K8s security controls'), 75)
        self.assertEqual(compute_match_score(['Go'], 'Google workloads'), 0)

    def test_source_fields_preserved_and_unwanted_fields_removed(self):
        source = tailor._source_context(master())
        self.assertEqual(source["name"], "TEST CANDIDATE")
        self.assertEqual(source["contact"], ["Bay Area | test@example.com"])
        self.assertEqual(source["education"], ["A.S. Computer Science & Engineering, Example College", "Certifications: AWS Certified Cloud Practitioner (CCP)"])
        self.assertEqual(source["languages"], "Languages: English (Fluent)")
        tailor._validate(candidate())

    def test_invalid_structures_and_duplicates_rejected(self):
        def summary(p): p.summary = "word " * 26
        def skills(p): p.technical_expertise.pop()
        def leadership(p): p.technical_expertise[2] = "Teamwork: Coordinate departments."
        def arqon(p): p.experience_bullets[0].pop(0)
        def ventera(p): p.experience_bullets[1].append("Extra bullet.")
        def project(p): p.experience_bullets[0][-1] = "Selected Project: Wrong - Delivered releases."
        def duplicate_tool(p): p.technical_expertise[0] += " Terraform automation."
        def duplicate_claim(p): p.experience_bullets[1][0] = p.experience_bullets[0][0]
        def alias(p):
            p.technical_expertise[0] += " Kubernetes orchestration."
            p.experience_bullets[0][0] += " K8s clusters."
        for mutate in (summary, skills, leadership, arqon, ventera, project, duplicate_tool, duplicate_claim, alias):
            with self.subTest(rule=mutate.__name__):
                payload = candidate()
                mutate(payload)
                with self.assertRaises(TailoringError):
                    tailor._validate(payload)

    def test_real_pdf_structure_summary_and_margins(self):
        context = tailor._context(tailor._source_context(master()), candidate())
        pdf = build_ats_pdf(context)
        self.assertEqual(len(PdfReader(io.BytesIO(pdf)).pages), 1)
        with pdfplumber.open(io.BytesIO(pdf)) as doc:
            page = doc.pages[0]
            text = page.extract_text()
            header = page.extract_text_lines()[0]
            self.assertAlmostEqual(header['chars'][0]['size'], 12.48, delta=0.03)
            self.assertIn("CORE SKILLS", text)
            self.assertIn("Work Authorization: United States Citizen (No sponsorship required)", text)
            self.assertNotIn("TECHNICAL EXPERTISE", text)
            self.assertNotIn("Tooling", text)
            lines = page.extract_text_lines()
            start = next(i for i, line in enumerate(lines) if line["text"] == "EXECUTIVE SUMMARY")
            end = next(i for i, line in enumerate(lines) if line["text"] == "CORE SKILLS")
            self.assertLessEqual(end - start - 1, 2)
            self.assertGreaterEqual(min(c['x0'] for c in page.chars), 35.5)
            self.assertLessEqual(max(c['x1'] for c in page.chars), 576.5)
            self.assertLessEqual(max(c['bottom'] for c in page.chars), 756.5)
            self.assertGreater(max(c['bottom'] for c in page.chars), 740)

    def test_design_system_colors_dividers_and_bullet_placement(self):
        """Reference typography and consistent hanging bullets are renderer-owned."""
        context = tailor._context(tailor._source_context(master()), candidate())
        html = render_cv_html(context)
        self.assertIn('#0043ce', html)
        self.assertNotIn('space-between', html)
        self.assertEqual(html.count('<li '), 14)
        self.assertEqual(html.count('class="skills-row"'), 3)
        self.assertEqual(html.count('class="selected-project"'), 2)
        self.assertIn('<strong>Selected Project:</strong> Release Automation System', html)
        slate_blue = (0.184, 0.329, 0.588)
        pdf = build_ats_pdf(context)
        with pdfplumber.open(io.BytesIO(pdf)) as doc:
            page = doc.pages[0]
            heading = next(l for l in page.extract_text_lines() if l['text'] == 'CORE SKILLS')
            self.assertEqual(tuple(round(v, 3) for v in heading['chars'][0]['non_stroking_color']), slate_blue)
            company = next(l for l in page.extract_text_lines() if l['text'].startswith('Arqon Consulting'))
            self.assertEqual(tuple(round(v, 3) for v in company['chars'][0]['non_stroking_color']), (0.122, 0.216, 0.388))
            self.assertAlmostEqual(heading['chars'][0]['size'], 9.96, delta=0.03)
            rules = [rect for rect in page.rects if rect['width'] > 500 and rect['height'] < 1]
            self.assertEqual(len(rules), 5)
            for rule in rules:
                self.assertAlmostEqual(rule['height'], 0.48, delta=0.25)
                self.assertAlmostEqual(rule['non_stroking_color'][0], 160 / 255, delta=0.003)

    def test_physical_summary_overflow_and_page_overflow_rejected(self):
        context = tailor._context(tailor._source_context(master()), candidate())
        long_summary = deepcopy(context)
        long_summary['summary'] = 'uncharacteristicallyelongatedword ' * 25
        with self.assertRaisesRegex(CVOverflowError, 'two physical lines'):
            build_ats_pdf(long_summary)
        context['experience'][0]['bullets'][0] = 'Long accomplishment describing production infrastructure. ' * 150
        with self.assertRaises(CVOverflowError):
            build_ats_pdf(context)


class FinalizedPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_job_list_uses_same_grounded_pipeline_as_email(self):
        expected = object()
        with patch.object(tailor, 'generate_tailored_result', new=AsyncMock(return_value=(expected, b'%PDF'))) as generate:
            result = await tailor_cv(master(), 'Cloud Engineer', 'Target Company', 'JD')
            self.assertIs(result, expected)
            generate.assert_awaited_once()
    async def test_jd_title_and_rulebook_reach_model_and_corrected_candidate_is_validated(self):
        invalid = candidate()
        invalid.summary = 'word ' * 26
        with patch.object(tailor, '_request_tailored_payload', new=AsyncMock(side_effect=[invalid, candidate()])) as request, \
                patch.object(tailor, 'build_ats_pdf', return_value=b'%PDF') as render:
            pdf = await tailor.generate_tailored_resume('Target job description', master(), 'Cloud Engineer')
            self.assertEqual(pdf, b'%PDF')
            self.assertIn('Target job description', request.call_args_list[0].args[1])
            self.assertIn('Cloud Engineer', request.call_args_list[0].args[1])
            self.assertEqual(request.call_args.kwargs['system_prompt'], tailor.RULEBOOK)
            self.assertEqual(request.await_count, 2)
            render.assert_called_once()
            self.assertTrue(render.call_args.args[0]['finalized'])

    async def test_provider_failure_is_not_retried(self):
        with patch.object(tailor, '_request_tailored_payload', new=AsyncMock(side_effect=LLMExecutionError('offline'))) as request:
            with self.assertRaises(LLMExecutionError):
                await tailor.generate_tailored_resume('JD', master())
            self.assertEqual(request.await_count, 1)

    async def test_overflow_retries_are_bounded(self):
        with patch.object(tailor, '_request_tailored_payload', new=AsyncMock(return_value=candidate())) as request, \
                patch.object(tailor, 'build_ats_pdf', side_effect=CVOverflowError('too long')):
            with self.assertRaisesRegex(TailoringError, 'three attempts'):
                await tailor.generate_tailored_resume('JD', master())
            self.assertEqual(request.await_count, 3)
