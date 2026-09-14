import io
import re
import pdfplumber
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.cv_tailor import _TailoredPayload, _reconstruct_tailored_text, template_context_from_text
from app.services.pdf_generator import CVOverflowError, build_ats_pdf, render_cv_html
from pypdf import PdfReader

CV = '''JANE DOE | CLOUD ENGINEER
Bay Area | jane@example.com

EXECUTIVE SUMMARY
Cloud engineer delivering reliable **AWS** systems.

TECHNICAL EXPERTISE
- Cloud: AWS, Azure

PROFESSIONAL EXPERIENCE
Engineer | Example | 2023 - Present
Production systems
- Automated deployments with Terraform.

EDUCATION
BSc Computer Science, Example University

ADDITIONAL
Languages: English
'''


class PDFGeneratorTests(unittest.TestCase):
    def test_model_role_changes_only_header_role(self):
        contact = 'Bay Area | jane@example.com'
        source = [{'name': 'Header', 'content': 'MUHAMMAD YUSUF | CLOUD ENGINEER\n' + contact}]
        text = _reconstruct_tailored_text(source, _TailoredPayload(role_title='Platform Engineer'))
        data = template_context_from_text(text)
        html = render_cv_html(data)
        self.assertEqual(data['role_title'], 'PLATFORM ENGINEER')
        self.assertEqual(data['header']['contact'], [contact])
        self.assertIn('MUHAMMAD YUSUF | PLATFORM ENGINEER | MULTI-CLOUD | TERRAFORM | CI/CD | CLOUD AUTOMATION', re.sub(r'<[^>]+>', '', html))
        self.assertNotIn('<h1', html)

    def test_template_escapes_content_and_preserves_bold(self):
        data = template_context_from_text(CV)
        data['summary'] = '**AWS** <img src="file:///secret"> & <script>alert(1)</script>'
        html = render_cv_html(data)
        self.assertIn('<strong>AWS</strong>', html)
        self.assertNotIn('<img', html)
        self.assertNotIn('<script', html)
        self.assertIn('&lt;script&gt;', html)
        self.assertEqual(data['header']['name'], 'JANE DOE | CLOUD ENGINEER')
        self.assertEqual(data['experience_bullets'], [['Automated deployments with Terraform.']])

    def test_render_single_page_and_preserve_all_sections(self):
        pdf = build_ats_pdf(template_context_from_text(CV))
        reader = PdfReader(io.BytesIO(pdf))
        self.assertEqual(len(reader.pages), 1)
        self.assertAlmostEqual(float(reader.pages[0].mediabox.width), 612, delta=1)
        text = reader.pages[0].extract_text()
        for expected in ('MUHAMMAD YUSUF', 'EXECUTIVE SUMMARY', 'TECHNICAL EXPERTISE',
                         'PROFESSIONAL EXPERIENCE', 'EDUCATION', 'ADDITIONAL',
                         'Example University', 'Languages: English'):
            self.assertIn(expected, text)

    def test_long_headline_stays_one_line_without_scaling_body(self):
        data = template_context_from_text(CV)
        data['role_title'] = 'DATA CENTER PLANT ENGINEER, MECHANICAL'
        with pdfplumber.open(io.BytesIO(build_ats_pdf(data))) as document:
            page = document.pages[0]
            line = page.extract_text_lines()[0]
            self.assertIn('CLOUD AUTOMATION', line['text'])
            self.assertIn(data['role_title'], line['text'])
            self.assertLessEqual(line['x1'], 576.1)
            heading = next(l for l in page.extract_text_lines() if l['text'] == 'EXECUTIVE SUMMARY')
            self.assertAlmostEqual(heading['chars'][0]['size'], 9.96, delta=0.03)

    def test_overflow_is_rejected_without_truncation(self):
        data = template_context_from_text(CV)
        data['experience_bullets'][0] = ['A long accomplishment describing production systems.'] * 120
        with self.assertRaises(CVOverflowError):
            build_ats_pdf(data)

    def test_rendered_design_matches_master_measurements(self):
        data = template_context_from_text(CV)
        with pdfplumber.open(io.BytesIO(build_ats_pdf(data))) as document:
            page = document.pages[0]
            lines = page.extract_text_lines()
            heading = next(line for line in lines if line['text'] == 'TECHNICAL EXPERTISE')
            self.assertTrue(all('Bold' not in char['fontname'] for char in heading['chars']))
            self.assertAlmostEqual(heading['chars'][0]['size'], 9.96, delta=0.03)
            self.assertEqual(tuple(round(v, 3) for v in heading['chars'][0]['non_stroking_color']),
                             (0.184, 0.329, 0.588))
            rules = [rect for rect in page.rects if rect['width'] > 500]
            self.assertTrue(rules)
            for rule in rules:
                self.assertAlmostEqual(rule['height'], 0.24, delta=0.01)
                self.assertAlmostEqual(rule['x0'], 36, delta=0.1)
                self.assertAlmostEqual(rule['x1'], 576, delta=0.3)
                self.assertAlmostEqual(rule['non_stroking_color'][0], 0.627, delta=0.002)
            self.assertTrue(any(0 < heading['top'] - rule['top'] < 13 for rule in rules))
            self.assertIn('1/1', page.extract_text())


if __name__ == '__main__':
    unittest.main()
