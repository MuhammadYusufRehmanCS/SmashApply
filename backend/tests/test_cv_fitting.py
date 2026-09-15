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
from app.routers.jobs import _run_tailor, download_cv
from app.services.cv_fitting import fit_tailored_cv
from app.services.cv_tailor import _TailoredPayload, _result_from_payload, TailoringError
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
        with patch('app.services.cv_fitting._request_tailored_payload', new_callable=AsyncMock,
                   return_value=self.payload) as request:
            result, pdf = await fit_tailored_cv(long_result, self.master, 'Systems Engineer', 'Example', 'Reliable services')
        self.assertEqual(len(PdfReader(io.BytesIO(pdf)).pages), 1)
        self.assertEqual(result.template_data['experience_bullets'], self.payload.experience_bullets)
        self.assertEqual(request.await_count, 1)
        self.assertIn('EXACT bullet', request.await_args.args[1])
        self.assertIn('Computer Science, Example College', result.text)

    async def test_fitting_pdf_does_not_call_model(self):
        with patch('app.services.cv_fitting._request_tailored_payload', new_callable=AsyncMock) as request:
            result, pdf = await fit_tailored_cv(self.result, self.master, 'Systems Engineer', 'Example', 'Services')
        request.assert_not_called()
        self.assertEqual(result.text, self.result.text)

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
