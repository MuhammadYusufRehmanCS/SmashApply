import io
import re
import pdfplumber
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.cv_tailor import _TailoredPayload, _reconstruct_tailored_text, template_context_from_text, short_role_title
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
        self.assertIn('MUHAMMAD YUSUF | PLATFORM ENGINEER', re.sub(r'<[^>]+>', '', html))
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
        for expected in ('MUHAMMAD YUSUF', 'EXECUTIVE SUMMARY', 'CORE SKILLS',
                         'PROFESSIONAL EXPERIENCE', 'EDUCATION', 'ADDITIONAL',
                         'Example University', 'Languages: English'):
            self.assertIn(expected, text)

    def test_long_header_shrinks_font_to_stay_on_one_line(self):
        data = template_context_from_text(CV)
        with pdfplumber.open(io.BytesIO(build_ats_pdf(data))) as document:
            line = document.pages[0].extract_text_lines()[0]
            self.assertIn('CLOUD AUTOMATION', line['text'])
            # A header that fits keeps the full 12.48pt size.
            self.assertTrue(all(abs(c['size'] - 12.48) < .03 for c in line['chars']))
        for title in ('Distinguished Engineer', 'DATA CENTER PLANT ENGINEER, MECHANICAL',
                      'Senior Cloud Platform & DevSecOps Engineer'):
            data['role_title'] = title
            with pdfplumber.open(io.BytesIO(build_ats_pdf(data))) as document:
                lines = document.pages[0].extract_text_lines()
                line = lines[0]
                # One line: the whole banner ends on the first line, inside the margin.
                self.assertIn('CLOUD AUTOMATION', line['text'])
                self.assertLessEqual(line['x1'], 576.5)
                # The complete (4-word, 32-character) title is kept; the font shrinks instead.
                from app.services.cv_tailor import short_role_title
                self.assertIn(short_role_title(title).upper(), line['text'])
                sizes = {round(c['size'], 2) for c in line['chars']}
                self.assertEqual(len(sizes), 1)
                self.assertLess(sizes.pop(), 12.45)
                self.assertGreaterEqual(min(c['size'] for c in line['chars']), 9 - 0.03)
                self.assertIn('Automated deployments with Terraform.', document.pages[0].extract_text())
            self.assertEqual(data['role_title'], title)

    def test_overflow_is_rejected_without_truncation(self):
        data = template_context_from_text(CV)
        data['experience_bullets'][0] = ['A long accomplishment describing production systems.'] * 120
        with self.assertRaises(CVOverflowError) as caught:
            build_ats_pdf(data)
        measurements = caught.exception.measurements
        self.assertGreater(measurements['content_height'], measurements['available_height'])
        self.assertAlmostEqual(measurements['available_height'], 704 * 4 / 3, delta=0.1)
        self.assertEqual(len([f for f in measurements['fields']
                              if f['path'].startswith('experience_bullets')]), 120)

    def test_rendered_design_matches_master_measurements(self):
        data = template_context_from_text(CV)
        with pdfplumber.open(io.BytesIO(build_ats_pdf(data))) as document:
            page = document.pages[0]
            lines = page.extract_text_lines()
            heading = next(line for line in lines if line['text'] == 'CORE SKILLS')
            self.assertTrue(all('Bold' not in char['fontname'] for char in heading['chars']))
            self.assertAlmostEqual(heading['chars'][0]['size'], 9.96, delta=0.03)
            self.assertEqual(tuple(round(v, 3) for v in heading['chars'][0]['non_stroking_color']),
                             (0.184, 0.329, 0.588))
            rules = [rect for rect in page.rects if rect['width'] > 500 and rect['height'] < 1]
            self.assertEqual(len(rules), 4)
            for rule in rules:
                self.assertAlmostEqual(rule['height'], 0.48, delta=0.25)
                self.assertAlmostEqual(rule['x0'], 36, delta=0.1)
                self.assertAlmostEqual(rule['x1'], 576, delta=0.1)
                self.assertAlmostEqual(rule['non_stroking_color'][0], 160 / 255, delta=0.003)
            self.assertTrue(any(0 < heading['top'] - rule['bottom'] < 12 for rule in rules))


if __name__ == '__main__':
    unittest.main()
