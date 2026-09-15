import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.job_scraper import (
    _has_us_location_signal,
    _order_for_persistence,
    job_dedupe_keys,
    scrape_role_names,
)


class JobScraperTests(unittest.TestCase):
    def test_remote_location_requires_us_fallback(self):
        self.assertTrue(_has_us_location_signal("Remote", "Candidates must reside in United States."))
        self.assertFalse(_has_us_location_signal("Remote", "Join us from anywhere."))

    def test_non_us_location_is_not_rescued_by_description(self):
        self.assertFalse(_has_us_location_signal("China", "US equal opportunity employer text."))
        self.assertFalse(_has_us_location_signal("Canada", "United States benefits are listed below."))

    def test_scrape_roles_include_generic_devops_queries(self):
        roles = scrape_role_names("Cloud Engineer")
        self.assertIn("Cloud Engineer", roles)
        self.assertIn("DevOps Engineer", roles)
        self.assertIn("AWS DevOps Engineer", roles)

    def test_dedupe_keys_include_normalized_url_and_company_title(self):
        keys = job_dedupe_keys(
            {
                "title": "Cloud Engineer",
                "company": "Acme Inc.",
                "job_url": "https://boards.greenhouse.io/acme/jobs/123?utm_source=x&gh_jid=123",
            }
        )

        self.assertIn(("url", "https://boards.greenhouse.io/acme/jobs/123?gh_jid=123"), keys)
        self.assertIn(("company_title", "acme", "cloud engineer"), keys)

    def test_persistence_order_puts_direct_sources_after_jobspy(self):
        ordered = _order_for_persistence(
            [
                {"site": "greenhouse", "company": "A"},
                {"site": "linkedin", "company": "B"},
                {"site": "builtin", "company": "C"},
                {"site": "indeed", "company": "D"},
            ]
        )

        self.assertEqual([job["site"] for job in ordered], ["linkedin", "indeed", "greenhouse", "builtin"])


class FocusedScrapeTests(unittest.IsolatedAsyncioTestCase):
    async def test_stops_at_15_and_does_not_query_more_sources(self):
        from unittest.mock import AsyncMock, patch
        from types import SimpleNamespace
        from app.services.job_scraper import scrape_for_roles
        jobs = [dict(title='Cloud Engineer', company=f'Company {i}', job_url=f'https://example.com/{i}',
                     description='AWS Terraform', site='handshake') for i in range(30)]
        with patch('app.services.job_scraper.get_settings', return_value=SimpleNamespace(job_source_list=['handshake','lever'])), \
             patch('app.services.job_scraper._scrape_handshake_sources', new_callable=AsyncMock, return_value=(jobs, [])), \
             patch('app.services.job_scraper._scrape_lever_sources', new_callable=AsyncMock) as later:
            results, errors = await scrape_for_roles('Cloud Engineer', 'Remote', master_text='AWS Azure Terraform CI/CD')
        self.assertEqual(len(results), 15)
        later.assert_not_called()

    async def test_skips_existing_and_unrelated_jobs_before_limit(self):
        from unittest.mock import AsyncMock, patch
        from types import SimpleNamespace
        from app.services.job_scraper import scrape_for_roles
        jobs = [dict(title='Cloud Engineer', company=f'Company {i}', job_url=f'https://example.com/{i}',
                     description='AWS Terraform', site='handshake') for i in range(20)]
        jobs.insert(0, dict(title='Receptionist', company='Other', job_url='https://example.com/no', description='', site='handshake'))
        seen = job_dedupe_keys(jobs[1])
        with patch('app.services.job_scraper.get_settings', return_value=SimpleNamespace(job_source_list=['handshake'])), \
             patch('app.services.job_scraper._scrape_handshake_sources', new_callable=AsyncMock, return_value=(jobs, [])):
            results, errors = await scrape_for_roles('Receptionist', 'Remote', master_text='AWS Azure Terraform CI/CD', existing_keys=seen)
        self.assertEqual(len(results), 15)
        self.assertTrue(all(j['title']=='Cloud Engineer' and j['company']!='Company 0' for j in results))

    def test_handshake_public_data_parser(self):
        import json
        from app.services.job_scraper import _handshake_jobs
        listing = dict(jobTitle='Junior Systems Administrator', employerName='Example',
                       publicUrl='https://app.joinhandshake.com/public/jobs/123',
                       parsedLocations=[dict(city='Boston', state='Massachusetts')], firstActiveAt='2026-09-10T00:00:00Z')
        html = '<script id="__NEXT_DATA__" type="application/json">'+json.dumps({'props':{'pageProps':{'jobs':[listing]}}})+'</script>'
        jobs = _handshake_jobs(html, [('Systems Administrator', True)])
        self.assertEqual(len(jobs),1)
        self.assertEqual(jobs[0]['company'],'Example')
        self.assertEqual(jobs[0]['site'],'handshake')

    def test_cv_roles_do_not_include_unrelated_primary(self):
        from app.services.job_scraper import cv_search_roles, cv_alignment_score
        roles = cv_search_roles('AWS Azure Terraform Docker Kubernetes CI/CD Linux Windows Server', 'Receptionist')
        self.assertIn('Cloud Engineer',roles)
        self.assertNotIn('Receptionist',roles)
        self.assertEqual(cv_alignment_score({'title':'Staff Cloud Engineer'},'AWS Azure', [('Cloud Engineer',True)]),0)


if __name__ == "__main__":
    unittest.main()
