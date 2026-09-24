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


def rewritten_candidate():
    payload = candidate()
    payload.summary = 'Cloud engineer delivering reliable infrastructure through automation, release engineering and service ownership, aligning technical execution with business needs and cross-team delivery priorities.'
    payload.core_skills = [
        'Cloud Architecture: Design resilient infrastructure and plan workload capacity around application needs, connecting service availability with reliable operations and consistent environment management for enterprise production systems.',
        'Delivery Engineering: Support release governance and production readiness through repeatable delivery workflows, automated verification and coordinated change practices that connect software engineering with dependable operational outcomes.',
        'Leadership & Cross-Functional Collaboration: Take technical ownership, coordinate engineering priorities and communicate operational requirements across teams, supporting shared delivery goals through clear accountability and practical collaboration.',
    ]
    payload.experience_bullets = [[
        "Engineered resilient production workloads through high-availability architecture, sustaining 99.9% uptime while aligning infrastructure design with service reliability needs and supporting dependable operations across the systems used by delivery teams.",
        "Accelerated software delivery through GitHub Actions workflows and SonarQube quality gates, bringing deployment times below 60 seconds while integrating automated release checks into the engineering process for production changes.",
        "Reduced recovery time by applying Ansible configuration management alongside Python and Bash maintenance scripts, making infrastructure operations repeatable and helping engineering teams restore dependable service through consistent system maintenance practices.",
        "Selected Project: Release Automation System - Established Jenkins delivery pipelines with complete artifact traceability, connecting build outputs to release workflows so engineering teams could track the software delivered into production environments.",
    ], [
        "Streamlined enterprise release delivery to improve deployment efficiency by 60%, reducing operational errors through repeatable delivery practices that helped teams coordinate software changes and maintain a dependable path into production systems.",
        "Supported container service readiness in 10-30 seconds by configuring scalable ingress routing, connecting application delivery needs with runtime availability so engineering teams could bring deployed services online reliably and support production workloads.",
        "Selected Project: Automated Infrastructure Provisioning - Established modular Terraform workflows for environment creation in 5-10 minutes, making infrastructure setup repeatable and helping delivery teams prepare consistent environments for their application deployment requirements.",
    ]]
    return payload


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

    def test_expanded_word_budgets_render_within_full_page(self):
        payload = candidate()
        payload.summary = rewritten_candidate().summary
        payload.experience_bullets[0][0] = (
            'Designed and governed high-availability cloud architectures across multi-region environments, '
            'ensuring 99.9% operational uptime for mission-critical enterprise production workloads '
            'while aligning service reliability and infrastructure operations with business needs and delivery priorities.')
        self.assertLessEqual(len(payload.summary.split()), 25)
        self.assertGreater(len(payload.experience_bullets[0][0].split()), 30)
        context = tailor._context(tailor._source_context(master()), payload)
        with pdfplumber.open(io.BytesIO(build_ats_pdf(context))) as document:
            self.assertEqual(len(document.pages), 1)
            lines = document.pages[0].extract_text_lines()
            summary_start = next(i for i, line in enumerate(lines) if line['text'] == 'EXECUTIVE SUMMARY')
            summary_end = next(i for i, line in enumerate(lines) if line['text'] == 'CORE SKILLS')
            self.assertLessEqual(summary_end - summary_start - 1, 2)
            self.assertLess(lines[-2]['bottom'], 740)
            self.assertLess(lines[-2]['bottom'], lines[-1]['top'])
        payload.experience_bullets[0][0] += ' extra word word word word word'
        with self.assertRaisesRegex(TailoringError, '35'):
            tailor._validate(payload)

    def test_copied_bullets_rejected_and_rewritten_bullets_accepted(self):
        with self.assertRaisesRegex(tailor.FinalizedValidationError, 'unchanged'):
            tailor._validate_active_rewrite(candidate(), master())
        tailor._validate_active_rewrite(rewritten_candidate(), master())
        self.assertEqual(__import__('app.services.cv_tailor', fromlist=['short_role_title']).short_role_title(
            'Senior Cloud Platform Infrastructure Operations Engineer'), 'Senior Cloud Platform Engineer')

    def test_underfilled_generation_reports_all_fields_without_padding(self):
        payload = rewritten_candidate()
        payload.summary = 'Cloud engineer supporting dependable production services.'
        payload.core_skills[0] = 'Cloud Architecture: Reliable service design.'
        with self.assertRaisesRegex(tailor.FinalizedValidationError, 'Insufficient technical detail') as caught:
            tailor._validate_active_rewrite(payload, master())
        self.assertIn('summary: target 20-25', str(caught.exception))
        self.assertIn('Core Skills 1: target 25-30', str(caught.exception))

    def test_jd_required_tools_still_cannot_repeat(self):
        payload = candidate()
        payload.summary = 'Cloud engineer building reliable AWS production services.'
        payload.core_skills[0] += ' Amazon Web Services and Terraform.'
        source = master().raw_text + ' AWS production services.'
        with self.assertRaises(tailor.DuplicateTechnologyError) as caught:
            tailor._validate(payload, source, 'AWS and Terraform infrastructure engineering')
        repairs = {r['technology']: r for r in caught.exception.repairs}
        self.assertEqual(repairs['Terraform']['keep_in'], 'experience_bullets.1.2')
        self.assertIn('core_skills.0', repairs['Terraform']['rewrite_without_name_or_alias'])
        self.assertEqual(repairs['AWS']['occurrences'], {'summary': 1, 'core_skills.0': 1})

    def test_aliases_and_role_title_share_the_strict_budget(self):
        payload = candidate()
        payload.role_title = 'Terraform Engineer'
        with self.assertRaises(tailor.DuplicateTechnologyError) as caught:
            tailor._validate(payload, master().raw_text, 'Terraform')
        self.assertIn('role_title', caught.exception.repairs[0]['rewrite_without_name_or_alias'])
        payload = candidate()
        payload.core_skills[0] += ' AWS and Amazon Web Services.'
        with self.assertRaises(tailor.DuplicateTechnologyError) as caught:
            tailor._validate(payload, master().raw_text, 'AWS')
        self.assertEqual(caught.exception.repairs[0]['occurrences']['core_skills.0'], 2)

    def test_partial_repairs_preserve_valid_edits_and_untouched_experience(self):
        payload = rewritten_candidate()
        error = tailor.FieldValidationError('repair needed', ['summary', 'core_skills.0'])
        plan = tailor._repair_plan(payload, error)
        changed_summary = payload.summary.replace('Cloud engineer delivering', 'Cloud engineer providing')
        repaired, errors = tailor._apply_field_repairs(payload, {
            'summary': changed_summary,
            'core_skills.0': payload.core_skills[0] + ' Terraform.',
        }, plan)
        self.assertEqual(errors, ['core_skills.0'])
        self.assertEqual(repaired.summary, changed_summary)
        self.assertEqual(repaired.core_skills, payload.core_skills)
        self.assertEqual(repaired.experience_bullets, payload.experience_bullets)

    def test_repair_feedback_keeps_exact_rejection_and_previous_output(self):
        payload = rewritten_candidate()
        error = tailor.FieldValidationError('repair', ['summary'])
        plan = tailor._repair_plan(payload, error)
        rejected_text = 'Cloud engineer supporting reliable operations.'
        _, rejected = tailor._apply_field_repairs(payload, {'summary': rejected_text}, plan)
        self.assertEqual(rejected, ['summary'])
        self.assertIn('Returned 5 words; required 20-25 words.', plan['summary']['last_rejection']['reasons'])
        next_plan = tailor._repair_plan(payload, error, plan)
        self.assertEqual(next_plan['summary']['last_rejection']['text'], rejected_text)
        tailor._apply_field_repairs(payload, {'summary': rejected_text}, next_plan)
        self.assertTrue(any('identical' in reason for reason in next_plan['summary']['last_rejection']['reasons']))

    def test_small_overflow_does_not_shorten_every_three_line_bullet(self):
        payload = rewritten_candidate()
        fields = [dict(path=f'experience_bullets.{i}.{j}', lines=3, line_height=16,
                       width=670, prefix_width=0, average_char_width=5.5)
                  for i, group in enumerate(payload.experience_bullets) for j, _ in enumerate(group)]
        error = CVOverflowError('page overflow', measurements=dict(
            content_height=950, available_height=939, fields=fields))
        plan = tailor._repair_plan(payload, error)
        self.assertEqual(len(plan), 1)
        spec = next(iter(plan.values()))
        self.assertEqual((spec['min_words'], spec['max_words']), (28, 35))
        self.assertIn('target_characters', spec)
        self.assertNotIn('max_characters', spec)
        self.assertEqual(spec['generation_word_count'], 28)
        next_plan = tailor._repair_plan(payload, error, plan)
        next_spec = next(iter(next_plan.values()))
        self.assertLess(next_spec['target_characters'], spec['target_characters'])
        self.assertEqual(next_spec['generation_word_count'], 28)

    def test_character_estimate_cannot_reject_valid_wording_before_pdf_check(self):
        payload = rewritten_candidate()
        path = 'experience_bullets.0.0'
        plan = tailor._repair_plan(payload, tailor.FieldValidationError('fit', [path]))
        plan[path]['target_characters'] = 224
        text = payload.experience_bullets[0][0]
        self.assertGreater(len(text), 224)
        repaired, rejected = tailor._apply_field_repairs(payload, {path: text}, plan)
        self.assertEqual(rejected, [])
        tailor._validate(repaired, master().raw_text)
        tailor._validate_active_rewrite(repaired, master())
        # The estimated target never overrides actual content rules.
        _, rejected = tailor._apply_field_repairs(payload, {path: 'too short'}, plan)
        self.assertEqual(rejected, [path])

    def test_new_core_skills_key_and_legacy_alias_share_one_schema(self):
        payload = candidate()
        data = payload.model_dump()
        self.assertIn('core_skills', data)
        self.assertNotIn('technical_expertise', data)
        self.assertEqual(_TailoredPayload.model_validate(data).core_skills, payload.core_skills)

    def test_title_and_skill_word_budgets_are_rejected(self):
        for edit in (
            lambda p: setattr(p, 'role_title', 'Senior Cloud Platform Infrastructure Security Operations Engineering Manager'),
            lambda p: p.core_skills.__setitem__(0, 'Cloud Architecture: ' + 'service ' * 31),
        ):
            payload = candidate()
            edit(payload)
            with self.assertRaises(TailoringError):
                tailor._validate(payload, master().raw_text)

    def test_scope_expansion_accepts_new_tools_standards_and_metrics(self):
        payload = candidate()
        payload.experience_bullets[0][0] = 'Architected NIST controls with Snowflake telemetry, governing 99.99% service uptime.'
        tailor._validate(payload, master().raw_text, 'NIST Snowflake')
        plan = tailor._repair_plan(payload, tailor.FieldValidationError('expand', ['core_skills.0']))
        self.assertNotIn('ArgoCD', plan['core_skills.0']['forbidden_terms'])
        self.assertIn('Snowflake', plan['core_skills.0']['forbidden_terms'])

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
        def summary(p): p.summary = "word " * 41
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
            self.assertLessEqual(end - start - 1, 3)
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
        self.assertNotIn('min-height:', html)
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
    def setUp(self):
        # Production defaults to one attempt; these tests exercise the repair loop itself.
        settings = tailor.get_settings().model_copy(update={"tailoring_max_attempts": 12})
        patcher = patch.object(tailor, 'get_settings', return_value=settings)
        patcher.start()
        self.addCleanup(patcher.stop)

    async def test_expanded_metric_does_not_trigger_source_membership_repair(self):
        good = rewritten_candidate()
        bad = good.model_copy(deep=True)
        bad.experience_bullets[0][0] = bad.experience_bullets[0][0].replace('99.9%', '99.999%')
        with patch.object(tailor, '_request_tailored_payload', new=AsyncMock(return_value=bad)) as request, \
                patch.object(tailor, 'request_field_repairs', new=AsyncMock(return_value={
                    'experience_bullets.0.0': good.experience_bullets[0][0]})) as repair, \
                patch.object(tailor, 'build_ats_pdf', return_value=b'%PDF'):
            await tailor.generate_tailored_result('Cloud services', master())
        request.assert_awaited_once()
        repair.assert_not_awaited()

    def test_misplaced_project_is_targeted_without_preserving_wrong_prefix(self):
        bad = rewritten_candidate()
        bad.experience_bullets[0][0] = 'Selected Project: ' + bad.experience_bullets[0][0]
        with self.assertRaises(tailor.FieldValidationError) as caught:
            tailor._validate(bad, master().raw_text)
        plan = tailor._repair_plan(bad, caught.exception)
        self.assertEqual(set(plan), {'experience_bullets.0.0'})
        self.assertEqual(plan['experience_bullets.0.0']['required_prefix'], '')

    async def test_overflow_repair_renders_one_page_without_character_rejections(self):
        original = rewritten_candidate()
        fixed = original.model_copy(deep=True)
        fixed.core_skills[1] = 'Delivery Engineering: Guide release checks and service readiness with clear change plans, repeatable build steps and shared review practices that help teams deliver reliable production changes.'
        fixed.core_skills[2] = 'Leadership & Cross-Functional Collaboration: Lead technical work, align team goals and share service needs across groups to support clear ownership, sound decisions and reliable software delivery.'
        fixed.experience_bullets[0][0] = 'Built resilient cloud systems to sustain 99.9% uptime for live workloads, linking high availability design to service needs so teams could run critical systems with fewer service disruptions.'
        fixed.experience_bullets[0][2] = 'Used Ansible with Python and Bash scripts to make system setup and upkeep repeatable, reduce recovery time and help teams restore services with clear, consistent steps during operational work.'
        fixed.experience_bullets[0][3] = 'Selected Project: Release Automation System - Built Jenkins release flows to track each artifact from build to deployment, giving teams a clear record of software changes across production stages.'
        fixed.experience_bullets[1][0] = 'Raised release delivery efficiency by 60% through repeatable workflows that reduced errors, helped teams manage software changes and made each deployment easier to carry out across complex enterprise systems.'
        fixed.experience_bullets[1][1] = 'Configured scalable ingress routes for container workloads to reach service readiness in 10-30 seconds, helping teams bring services online and meet runtime needs with reliable traffic paths for deployed applications.'
        async def repair(settings, fields, context, feedback):
            return {path: tailor._editable_fields(fixed)[path] for path in fields}
        with patch.object(tailor, '_request_tailored_payload', new=AsyncMock(return_value=original)) as request, \
                patch.object(tailor, 'request_field_repairs', new=AsyncMock(side_effect=repair)) as repair_call:
            result, pdf = await tailor.generate_tailored_result('Reliable cloud services', master())
        request.assert_awaited_once()
        repair_call.assert_awaited_once()
        self.assertEqual(len(PdfReader(io.BytesIO(pdf)).pages), 1)
        self.assertIn(original.experience_bullets[1][2], result.text)
        self.assertGreater(len(original.experience_bullets[1][2]), 222)

    async def test_job_list_uses_same_grounded_pipeline_as_email(self):
        expected = object()
        with patch.object(tailor, 'generate_tailored_result', new=AsyncMock(return_value=(expected, b'%PDF'))) as generate:
            result = await tailor_cv(master(), 'Cloud Engineer', 'Target Company', 'JD')
            self.assertIs(result, expected)
            generate.assert_awaited_once()
    async def test_jd_title_and_rulebook_reach_model_and_corrected_candidate_is_validated(self):
        invalid = rewritten_candidate()
        invalid.summary = 'word ' * 41
        with patch.object(tailor, '_request_tailored_payload', new=AsyncMock(return_value=invalid)) as request, \
                patch.object(tailor, 'request_field_repairs', new=AsyncMock(
                    return_value={'summary': rewritten_candidate().summary})) as repair, \
                patch.object(tailor, 'build_ats_pdf', return_value=b'%PDF') as render:
            pdf = await tailor.generate_tailored_resume('Target job description', master(), 'Cloud Engineer')
        self.assertEqual(pdf, b'%PDF')
        self.assertIn('Target job description', request.call_args.args[1])
        self.assertEqual(request.call_args.kwargs['system_prompt'], tailor.RULEBOOK)
        self.assertEqual(request.await_count, 1)
        self.assertEqual(set(repair.call_args.args[1]), {'summary'})
        render.assert_called_once()

    async def test_duplicate_retry_repairs_only_failing_field_and_keeps_rules(self):
        invalid = rewritten_candidate()
        invalid.core_skills[0] += ' Terraform.'
        with patch.object(tailor, '_request_tailored_payload', new=AsyncMock(return_value=invalid)) as request, \
                patch.object(tailor, 'request_field_repairs', new=AsyncMock(
                    return_value={'core_skills.0': rewritten_candidate().core_skills[0]})) as repair, \
                patch.object(tailor, 'build_ats_pdf', return_value=b'%PDF') as render:
            await tailor.generate_tailored_resume('Terraform engineering', master(), 'Cloud Engineer')
        self.assertEqual(request.await_count, 1)
        plan = repair.call_args.args[1]
        self.assertEqual(set(plan), {'core_skills.0'})
        self.assertIn('Terraform', plan['core_skills.0']['forbidden_terms'])
        self.assertEqual(plan['core_skills.0']['min_words'], 25)
        self.assertEqual(render.call_args.args[0]['experience'][1]['bullets'], invalid.experience_bullets[1])

    async def test_repair_continues_past_three_attempts_without_regenerating_good_fields(self):
        invalid = rewritten_candidate()
        invalid.core_skills[0] += ' Terraform.'
        responses = [{'core_skills.0': invalid.core_skills[0]}] * 3 + [
            {'core_skills.0': rewritten_candidate().core_skills[0]}]
        with patch.object(tailor, '_request_tailored_payload', new=AsyncMock(return_value=invalid)) as request, \
                patch.object(tailor, 'request_field_repairs', new=AsyncMock(side_effect=responses)) as repair, \
                patch.object(tailor, 'build_ats_pdf', return_value=b'%PDF'):
            result, pdf = await tailor.generate_tailored_result('Terraform engineering', master())
        self.assertEqual(pdf, b'%PDF')
        self.assertEqual(request.await_count, 1)
        self.assertEqual(repair.await_count, 4)
        self.assertIn(invalid.summary, result.text)

    async def test_invalid_patch_retries_same_fields_without_losing_candidate(self):
        invalid = rewritten_candidate()
        invalid.core_skills[0] += ' Terraform.'
        with patch.object(tailor, '_request_tailored_payload', new=AsyncMock(return_value=invalid)) as request, \
                patch.object(tailor, 'request_field_repairs', new=AsyncMock(side_effect=[
                    tailor.PayloadFormatError('Invalid patch JSON'),
                    {'core_skills.0': rewritten_candidate().core_skills[0]}])) as repair, \
                patch.object(tailor, 'build_ats_pdf', return_value=b'%PDF'):
            await tailor.generate_tailored_resume('Terraform engineering', master())
        self.assertEqual(request.await_count, 1)
        self.assertEqual(repair.await_count, 2)
        self.assertEqual(repair.call_args_list[0].args[1], repair.call_args_list[1].args[1])

    async def test_provider_failure_is_not_retried(self):
        with patch.object(tailor, '_request_tailored_payload', new=AsyncMock(side_effect=LLMExecutionError('offline'))) as request:
            with self.assertRaises(LLMExecutionError):
                await tailor.generate_tailored_resume('JD', master())
            self.assertEqual(request.await_count, 1)

    async def test_overflow_retries_are_bounded(self):
        from app.config import Settings
        payload = rewritten_candidate()
        async def replacements(settings, fields, *args):
            return {path: tailor._editable_fields(payload)[path] for path in fields}
        with patch.object(tailor, 'get_settings', return_value=Settings(_env_file=None, tailoring_max_attempts=5)), \
                patch.object(tailor, '_request_tailored_payload', new=AsyncMock(return_value=payload)) as request, \
                patch.object(tailor, 'request_field_repairs', new=AsyncMock(side_effect=replacements)) as repair, \
                patch.object(tailor, 'build_ats_pdf', side_effect=CVOverflowError('too long')):
            with self.assertRaisesRegex(TailoringError, '5 attempts'):
                await tailor.generate_tailored_resume('JD', master())
        self.assertEqual(request.await_count, 1)
        self.assertEqual(repair.await_count, 4)

    async def test_deadline_stops_generation(self):
        import asyncio
        settings = SimpleNamespace(tailoring_max_attempts=12, tailoring_timeout_seconds=.01)
        async def slow(*args, **kwargs):
            await asyncio.sleep(2)
        with patch.object(tailor, 'get_settings', return_value=settings), \
                patch.object(tailor, '_request_tailored_payload', new=AsyncMock(side_effect=slow)):
            with self.assertRaisesRegex(TailoringError, 'seconds'):
                await tailor.generate_tailored_resume('JD', master())
