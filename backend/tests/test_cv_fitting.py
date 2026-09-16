import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fastapi import HTTPException
from pypdf import PdfReader
from app.models import MasterCV
from app.routers.jobs import _run_tailor, download_cv, _tailoring_failure_detail
from app.services.cv_fitting import fit_tailored_cv, _wording_limits
from app.services.cv_tailor import _TailoredPayload, _result_from_payload, TailoringError, LLMExecutionError
from app.services.pdf_generator import CVOverflowError


class FitTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.sections = [
            {'name': 'Header', 'content': 'MUHAMMAD YUSUF | SYSTEM ENGINEER\nBay Area, CA'},
            {'name': 'Executive Summary', 'content': 'Original summary.'},
            {'name': 'Professional Experience', 'content':
             'Cloud Engineer | Example | 2023 - Present\n'
             '- Designed and governed high-availability AWS architectures for production workloads.\n'
             '- Automated infrastructure with Terraform and Python to improve reliability.'},
            {'name': 'Education', 'content': '- Computer Science, Example College'},
        ]
        self.master = MasterCV(sections_json=json.dumps(self.sections), raw_text='Master CV', layout_json='{}')
        self.payload = _TailoredPayload(summary='Engineer supporting reliable production services.',
            experience_bullets=[[
                'Kept production services available by implementing resilient AWS systems and reviewing operational risks.',
                'Made service recovery repeatable through reusable provisioning modules and Python operational scripts.',
            ]])
        self.result = _result_from_payload(self.sections, self.payload, cacheable=True)
        self.job = Mock(id=1, title='Systems Engineer', company='Example', description='Reliable services',
                        tailored_cv=self.result.text, tailored_keywords='AWS, Python')

    async def test_real_overflow_is_rewritten_then_rendered_one_page(self):
        long_payload = self.payload.model_copy(deep=True)
        long_payload.experience_bullets[0] = [bullet * 50 for bullet in self.payload.experience_bullets[0]]
        long_result = _result_from_payload(self.sections, long_payload, cacheable=True)
        with patch('app.services.cv_fitting.request_wording_replacements', new_callable=AsyncMock,
                   return_value={f'experience_bullets.0.{i}': text for i, text in enumerate(self.payload.experience_bullets[0])}) as request:
            result, pdf = await fit_tailored_cv(long_result, self.master, 'Systems Engineer', 'Example', 'Reliable services')
        self.assertEqual(len(PdfReader(io.BytesIO(pdf)).pages), 1)
        self.assertEqual(result.template_data['experience_bullets'], self.payload.experience_bullets)
        self.assertEqual(request.await_count, 1)
        self.assertIn('experience_bullets.0.0', request.await_args.args[1])
        self.assertIn('Computer Science, Example College', result.text)

    async def test_fitting_pdf_does_not_call_model(self):
        with patch('app.services.cv_fitting._request_tailored_payload', new_callable=AsyncMock) as request:
            result, pdf = await fit_tailored_cv(self.result, self.master, 'Systems Engineer', 'Example', 'Services')
        request.assert_not_called()
        self.assertEqual(result.text, self.result.text)

    async def test_rejected_revision_keeps_validation_reason_and_does_not_rerender_old_cv(self):
        invalid = self.payload.model_copy(deep=True)
        invalid.experience_bullets = [[]]
        with patch('app.services.cv_fitting.build_ats_pdf', side_effect=CVOverflowError('too long')) as render, \
             patch('app.services.cv_fitting._request_tailored_payload', new_callable=AsyncMock,
                   return_value=invalid) as request:
            with self.assertRaises(TailoringError) as caught:
                await fit_tailored_cv(self.result, self.master, 'Engineer', 'Example', 'Services')
        self.assertEqual(render.call_count, 1)
        self.assertEqual(request.await_count, 3)
        self.assertIn('wrong bullet count', str(caught.exception.__cause__))
        self.assertIn('wrong bullet count', request.await_args.args[1])
        self.assertIn('wrong bullet count', _tailoring_failure_detail(caught.exception))

    async def test_provider_failure_is_not_retried_as_a_wording_problem(self):
        failure = LLMExecutionError('Private provider body')
        with patch('app.services.cv_fitting.build_ats_pdf', side_effect=CVOverflowError('too long')), \
             patch('app.services.cv_fitting._request_tailored_payload', new_callable=AsyncMock,
                   side_effect=failure) as request:
            with self.assertRaises(LLMExecutionError):
                await fit_tailored_cv(self.result, self.master, 'Engineer', 'Example', 'Services')
        self.assertEqual(request.await_count, 1)
        self.assertIn('OpenAI', _tailoring_failure_detail(failure))
        self.assertNotIn('Private provider body', _tailoring_failure_detail(failure))

    def test_failure_details_distinguish_overflow_from_quota(self):
        self.assertIn('exceeds one page', _tailoring_failure_detail(CVOverflowError('too long')))
        provider = RuntimeError('private')
        provider.status_code = 429
        failure = LLMExecutionError('request failed')
        failure.__cause__ = provider
        self.assertIn('quota', _tailoring_failure_detail(failure))

    def test_measured_limits_shorten_skills_before_experience(self):
        fields = [dict(path=path, lines=lines, characters=chars, width=670,
                       prefix_width=prefix, average_char_width=5.5, line_height=16.32)
                  for path, lines, chars, prefix in [
                      ('summary', 1, 100, 0), ('technical_expertise.0', 3, 260, 120),
                      ('experience_bullets.0.0', 2, 160, 0),
                      ('experience_bullets.0.1', 2, 170, 0)]]
        measurements = dict(available_height=600, fixed_height=498, fields=fields)
        limits = _wording_limits(measurements)
        self.assertEqual(limits['technical_expertise.0']['lines'], 1)
        self.assertEqual(limits['experience_bullets.0.0']['lines'], 2)
        self.assertLess(limits['technical_expertise.0']['max_characters'], 100)
        self.assertLessEqual(sum(rule['lines'] * 16.32 for rule in limits.values()), 99)

    async def test_measured_repair_cannot_expand_already_fitting_fields(self):
        field = dict(path='experience_bullets.0.0', lines=3, characters=220, width=670,
                     prefix_width=0, average_char_width=5.5, line_height=16.32)
        measurements = dict(available_height=600, content_height=614, fixed_height=565, fields=[field])
        revision = self.payload.model_copy(deep=True)
        revision.summary = 'Unrequested expansion ' * 100
        revision.experience_bullets[0][0] = 'Built Terraform recovery modules to restore failed services.'
        revision.experience_bullets[0][1] = 'Unrequested expansion ' * 100
        with patch('app.services.cv_fitting.build_ats_pdf', side_effect=[
                CVOverflowError('too long', measurements=measurements), b'%PDF-test']), \
             patch('app.services.cv_fitting.request_wording_replacements', new_callable=AsyncMock,
                   return_value={'experience_bullets.0.0': revision.experience_bullets[0][0],
                                 'summary': revision.summary}) as request:
            result, _ = await fit_tailored_cv(self.result, self.master, 'Engineer', 'Example', 'Services')
        self.assertEqual(result.template_data['summary'], self.payload.summary)
        self.assertEqual(result.template_data['experience_bullets'][0][1], self.payload.experience_bullets[0][1])
        self.assertEqual(result.template_data['experience_bullets'][0][0], revision.experience_bullets[0][0])
        self.assertIn('max_characters', request.await_args.args[1]['experience_bullets.0.0'])

    def test_print_fragmentation_still_reclaims_a_line(self):
        fields = [dict(path='summary', lines=3, characters=250, width=670,
                       prefix_width=0, average_char_width=5.5, line_height=16.32)]
        limits = _wording_limits(dict(available_height=600, fixed_height=400, fields=fields))
        self.assertEqual(limits['summary']['lines'], 2)

    async def test_shortening_request_includes_category_label_with_tools(self):
        self.sections.insert(2, {'name': 'Technical Expertise', 'content': '- Cloud & Infrastructure: AWS, Azure'})
        self.master.sections_json = json.dumps(self.sections)
        self.payload.technical_expertise = ['AWS, Azure, GCP, VPC, Data Platform']
        candidate = _result_from_payload(self.sections, self.payload, cacheable=True)
        field = dict(path='technical_expertise.0', lines=3, characters=200, width=670,
                     prefix_width=120, average_char_width=5.5, line_height=16.32)
        measurement = dict(available_height=600, content_height=614, fixed_height=565, fields=[field])
        with patch('app.services.cv_fitting.build_ats_pdf', side_effect=[
                CVOverflowError('too long', measurements=measurement), b'%PDF-test']), \
             patch('app.services.cv_fitting.request_wording_replacements', new_callable=AsyncMock,
                   return_value={'technical_expertise.0': 'AWS, Azure, Data Platform'}) as request:
            await fit_tailored_cv(candidate, self.master, 'Engineer', 'Example', 'Services')
        spec = request.await_args.args[1]['technical_expertise.0']
        self.assertEqual(spec['category_label'], 'Cloud & Infrastructure')
        self.assertEqual(spec['text'], self.payload.technical_expertise[0])

    def test_stalled_overflow_strictly_reduces_the_previous_character_budget(self):
        field = dict(path='summary', lines=3, characters=250, width=670,
                     prefix_width=0, average_char_width=5.5, line_height=16.32)
        measurements = dict(available_height=600, fixed_height=565, fields=[field])
        first = _wording_limits(measurements)
        second = _wording_limits(measurements, first)
        third = _wording_limits(measurements, second)
        self.assertLess(second['summary']['max_characters'], first['summary']['max_characters'])
        self.assertLess(third['summary']['max_characters'], second['summary']['max_characters'])

    async def test_failed_fit_does_not_overwrite_cache(self):
        db = Mock()
        self.job.tailored_cv = None
        with patch('app.routers.jobs.fit_tailored_cv', new_callable=AsyncMock,
                   side_effect=TailoringError('Still too long')):
            with self.assertRaises(HTTPException):
                await _run_tailor(self.job, self.master, db, candidate=self.result)
        db.commit.assert_not_called()
        self.assertIsNone(self.job.tailored_cv)

    async def test_failed_refresh_returns_previous_valid_resume(self):
        db = Mock()
        with patch('app.routers.jobs.fit_tailored_cv', new_callable=AsyncMock,
                   side_effect=TailoringError('Revision unavailable')):
            result = await _run_tailor(self.job, self.master, db, candidate=self.result)
        self.assertEqual(result.text, self.result.text)
        self.assertTrue(result.used_fallback)
        self.assertFalse(result.cacheable)
        db.commit.assert_not_called()

    async def test_cached_overflow_runs_revision_instead_of_422(self):
        db = Mock()
        db.get.return_value = self.job
        with patch('app.routers.jobs._get_master_cv_or_400', return_value=self.master), \
             patch('app.routers.jobs._has_cached_tailoring', return_value=True), \
             patch('app.routers.jobs.build_ats_pdf', side_effect=[CVOverflowError('too long'), b'%PDF-test']), \
             patch('app.routers.jobs._run_tailor', new_callable=AsyncMock, return_value=self.result) as rewrite:
            response = await download_cv(1, db)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.body, b'%PDF-test')
        self.assertEqual(rewrite.await_args.kwargs['candidate'].text, self.result.text)


if __name__ == '__main__':
    unittest.main()
