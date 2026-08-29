"""Cloud/DevOps role taxonomy for the job search engine.

Every scrape queries PRIMARY_ROLE_DEFAULT (or whatever the caller passes in)
plus every title in ALIGNED_ROLES, so a single search surfaces the full
adjacent-role ecosystem instead of just literal title matches.
"""

PRIMARY_ROLE_DEFAULT = "Cloud Engineer"

ALIGNED_ROLES: list[str] = [
    "Site Reliability Engineer",
    "Platform Engineer",
    "Infrastructure Engineer",
    "Cloud Architect",
    "DevSecOps Engineer",
    "Build & Release Engineer",
    "Systems Development Engineer",
    "Cloud Systems Administrator",
    "Kubernetes Engineer",
    "Cloud Automation Engineer",
]
