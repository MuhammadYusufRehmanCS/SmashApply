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


if __name__ == "__main__":
    unittest.main()
