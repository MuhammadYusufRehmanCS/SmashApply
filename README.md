# SmashApply

Local Cloud/DevOps job-application engine: live-scrapes job boards for a role plus 10 aligned
titles, tailors your Master CV's wording to each posting with OpenAI GPT-4o, and generates
an ATS-friendly PDF that mirrors your Master CV's original layout — an OpenAI API key is required; no managed
databases.

## Stack

- **Backend:** Python / FastAPI / SQLAlchemy / SQLite
- **Layout parsing:** `pypdf` (text) + `pdfplumber` (font/position metadata)
- **Scraping:** `python-jobspy` against LinkedIn, Indeed, Glassdoor, ZipRecruiter
- **Tailoring:** OpenAI GPT-4o (`gpt-4o`) with strict structured JSON output
- **PDF generation:** Jinja2 HTML/CSS rendered by Playwright Chromium (fixed one-page template)
- **Frontend:** Next.js (App Router) / Tailwind CSS
- **Infra:** `docker-compose.yml` for one-command startup

## Project structure

```
smashapply/
├── backend/
│   ├── app/
│   │   ├── main.py            # FastAPI app + CORS + router registration
│   │   ├── config.py          # env-driven settings
│   │   ├── database.py        # SQLAlchemy engine/session
│   │   ├── models.py          # MasterCV, Job ORM models
│   │   ├── schemas.py         # Pydantic request/response models
│   │   ├── roles.py           # primary role + 10 aligned Cloud/DevOps titles
│   │   ├── routers/
│   │   │   ├── cv.py          # master CV upload + layout parsing
│   │   │   └── jobs.py        # scrape / list / tailor / download-cv
│   │   └── services/
│   │       ├── cv_layout.py       # pdfplumber+pypdf layout profile extraction
│   │       ├── text_sections.py   # shared heading/bullet detection heuristics
│   │       ├── job_scraper.py     # jobspy scraping across 11 role titles
│   │       ├── cv_tailor.py       # OpenAI-backed keyword extraction + rewrite
│   │       └── pdf_generator.py   # Jinja2 + Chromium single-page PDF
│   ├── requirements.txt
│   ├── Dockerfile
│   └── .env.example
├── frontend/
│   ├── app/                   # page.tsx (dashboard), layout.tsx, globals.css
│   ├── components/            # ui/* primitives (Button, Card, Badge)
│   ├── lib/                   # api.ts client, utils.ts
│   ├── types/job.ts
│   ├── package.json
│   └── Dockerfile
└── docker-compose.yml
```

## Data model (SQLite)

- **`master_cv`**: single-row table with `raw_text`, `sections_json` (ordered section blocks), and
  `layout_json` (font family/sizes, line spacing, margins, column count, section order).
- **`jobs`**: `title`, `company`, `location`, `job_url`, `site`, `description`, `role_category`
  (which of the 11 titles surfaced it), `is_primary_role`, plus `tailored_cv` /
  `tailored_keywords` once tailored. Deduplicated on `(title, company, job_url)`.

## Getting started (local, no Docker)

### 1. Backend

```bash
cd backend
python -m venv .venv
./.venv/Scripts/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m playwright install chromium
cp .env.example .env
uvicorn app.main:app --reload
```

API serves on `http://127.0.0.1:8000` (docs at `/docs`). SQLite data lives in
`backend/data/smashapply.db`, created automatically on first run.

**OpenAI setup:** set `OPENAI_API_KEY` in `backend/.env`. The default model is `gpt-4o`, configurable through `OPENAI_MODEL`. Restart the backend after changing these settings. API usage is billed separately; your project must have access to the selected model. Existing saved CVs must be tailored again to use the new model.

### 2. Frontend

```bash
cd frontend
npm install
cp .env.local.example .env.local
npm run dev
```

Dashboard serves on `http://localhost:3000`.

### 3. Use it

1. Upload your Master CV (`.pdf`) — the layout parse preview shows the detected section order,
   font, and column structure.
2. Set a Job Title (defaults to "Cloud Engineer / DevOps Engineer") and Location (defaults to
   "Remote"), click **Scrape Jobs**. This queries the primary title plus the 10 aligned Cloud/DevOps
   titles in `app/roles.py` across all four job boards.
3. Click **Tailor & Download CV** on any row: OpenAI extracts the job's keywords, rewrites your
   CV's bullets to mirror them (facts unchanged), and a PDF matching your Master CV's layout
   downloads automatically.

## Getting started (Docker)

```bash
cp backend/.env.example backend/.env
docker compose up --build
```

Frontend: `http://localhost:3000`. Backend: `http://localhost:8000`.

> If Ollama runs on your host machine (not in Docker), point `OLLAMA_BASE_URL` in `backend/.env`
> at `http://host.docker.internal:11434` instead of `localhost`.

## API overview

### Email referrals (local desktop)

Install the updated `backend/requirements.txt`, enable the Gmail API in your Google
Cloud project, configure the OAuth consent screen (add your account as a test user
if the app is in testing), and download a **Desktop app** OAuth client as
`credentials.json` in the repository root. `credentials.json.json` is also accepted
as a fallback for Windows downloads with a duplicated extension.

Upload your master CV, then click **Fetch Latest Email Referral & Tailor Resume**.
On first use, complete Google consent in the browser on the backend computer.
The app saves and refreshes `token.json` locally. This desktop callback flow is for
a locally running backend, not a remote or container-hosted OAuth deployment.

The endpoint `POST /api/inbox/tailor-latest` finds the latest unread inbox message
matching `is:unread (referral OR "job description" OR opportunity OR role)`, extracts
plain text or parsed HTML from nested MIME bodies, and runs the existing OpenAI
tailoring and PDF fitting pipeline. Plain text is preferred in multipart alternatives.
File attachments are skipped; attachment-backed inline text/HTML bodies are fetched.
The resulting PDF appears in an embedded preview with a download link.

The UI first calls `POST /api/inbox/extract-latest`, which uses local regex parsing
for `Job Title ::`, `Duration ::`, and `Job Summary:` / `Job Description:`. It returns
`job_title`, `sender` (from the email From header), `duration`, and `jd_text`.
The JD includes the first summary/description heading through the end of the body;
unstructured emails retain their full body, and absent metadata has explicit fallbacks.
No LLM is called during extraction. The button displays the extracted title and sender
before posting that metadata to `POST /api/inbox/tailor` for resume generation.
Both stages reject short JDs before generation. `/tailor-latest` remains available
as a combined endpoint for existing clients.

Gmail authorization uses `gmail.modify`. Existing read-only tokens automatically
trigger new consent. Each fetched message is marked read before returning, including
empty or malformed bodies. Empty or fewer-than-50-character JDs return HTTP 422;
an empty inbox also returns 422 with a distinct message. If tailoring fails, mark
that email unread in Gmail to retry it. Local processing history is no longer used.
OAuth files remain ignored by Git and Docker. Run one backend worker for this local
workflow; concurrent requests within the worker return a clear busy response.

| Method | Path                          | Purpose                                              |
|--------|--------------------------------|-------------------------------------------------------|
| POST   | `/api/cv/upload`              | Upload Master CV (.pdf); parses text + layout profile |
| GET    | `/api/cv`                     | Fetch current master CV + layout profile              |
| POST   | `/api/jobs/scrape`             | Scrape primary role + 10 aligned titles, dedupe, save  |
| GET    | `/api/jobs`                    | List scraped jobs                                      |
| POST   | `/api/jobs/{id}/tailor`        | OpenAI: extract keywords, rewrite CV bullets           |
| GET    | `/api/jobs/{id}/download-cv`   | Generate + download the tailored ATS PDF               |

## Known limitations

### Finalized resume contract

`app/services/tailor.py` uses `finalized_cv.html` for both email and job-list workflows when the master contains Arqon Consulting and Ventera Group. Upload the
finalized Master CV containing Arqon Consulting and Ventera Group, the A.S. degree,
a Certifications line, and Languages. The renderer preserves these source facts,
removes education dates, restores the approved fixed header banner, and fixes work authorization
to `United States Citizen (No sponsorship required)`.

Generation requires a maximum 40-word, three-physical-line summary; exactly three
Core Skills bullets (leadership/collaboration last); four Arqon and three Ventera
bullets with the prescribed final project prefixes. The model is instructed not to
repeat claims. Local checks reject exact/near-duplicate prose, repeated percentage
metrics, and repeated technology names/aliases across editable text, including
a shared technology/alias vocabulary. Arbitrary semantic paraphrases are not guaranteed
to be detected by those local heuristics. Immutable certification names are preserved.
Tools are strictly single-use across editable fields, including the role title.
Aliases share the same identity. JD relevance and ATS density do not create an
exception. Duplicate rejections give the model exact field paths and prioritize
retaining the Selected Project mention. Immutable banners/certifications are excluded.
Repeated accomplishments remain prohibited. Technical scope and metric expansion is enabled.

Every PDF entrypoint uses the same `finalized_cv.html` shell (`cv_template.html`
is only a compatibility include). It uses US Letter with half-inch margins,
The attached reference controls typography: Calibri 9.96pt body and section
headings, a uniform 12.48pt name/role/banner, blue section titles (#2f5496),
bright blue header tags and dark blue employer headings. Thin gray dividers
appear before sections and the second employer. Normal paragraph flow uses
hanging bullets and no stretched page gaps. Employer pipe separators are preserved.
A failed regeneration returns an explicit fallback warning; the UI does not
automatically download the saved resume as though it were a new tailored result.

Generation targets 20-25 summary words, 25-30 words per Core Skills category,
and 28-35 words per experience bullet. Local validation rejects underfilled
fields and copied/minimally edited experience before accepting a generated CV.
Prompts allow JD-driven workstreams, tools, frameworks and quantified outcomes beyond the master.

Core Skills, experience, education and additional information use consistent
hanging bullets. Only the Selected Project label is bold, followed by the existing
project wording. The renderer retains the existing structural validation. The LLM
supplies JSON text only; all layout, colors, identity and HTML are application-owned.

Both workflows share scope-expanding generation and structural validation. Prompts prioritize exact
JD terminology and senior ownership. Tools, standards and numbers are not checked for
membership in the master. Employers, dates and credentials remain immutable.
`core_skills` has `minItems = maxItems = 3`;
`technical_expertise` is accepted for backward compatibility. Summary remains capped
at 25 words/two physical lines, Core Skills rows at 30 words, all experience bullets at
35 words (target 28-35 words, roughly two lines). Invalid output enters a targeted field-repair loop. Valid fields remain unchanged;
all structure, duplication, word-budget and physical-page checks still run.
`TAILORING_MAX_ATTEMPTS` defaults to 12 total model calls (maximum 30), and
`TAILORING_TIMEOUT_SECONDS` defaults to 600 seconds. The first limit reached stops
processing without saving invalid content; API authentication/connection failures stop immediately.
Duplicate repairs forbid tools assigned to other fields; density repairs expand only
underfilled fields. Measured overflow repairs reclaim only the needed rendered
lines, preserving three-line experience bullets where space permits. Character
estimates guide rewriting; the unchanged PDF renderer enforces actual page fit,
while word ranges, prefixes and deduplication remain mandatory.
Temporary OpenAI 429 rate limits honor Retry-After (or the provider's requested
wait) and use bounded backoff. The finalized request shares at most three extra
rate-limit retries across generation and repairs, within the same wall-clock deadline.
Insufficient quota is never retried and receives a separate billing/quota message.
Rejected field repairs include exact word/character counts, forbidden terms and
the last rejected text in the next correction request.
These bounds prevent indefinite API spend. No padding or fabricated workstreams are allowed.

The inspected file was `smesh_cloudeng.pdf`; `smesh_cloudeng_5.pdf` was not available.
The explicit current design rules take precedence over differences in that PDF
(including its multiple blue shades, older font sizes, and bullet icons).

Match score is unique JD-keyword coverage, recognizing aliases such as AWS/Amazon
Web Services and K8s/Kubernetes. Repetition gives no extra credit. Unmet requirements
remain in the denominator, so a higher score is not guaranteed and must not be
achieved by inventing qualifications. Regenerate a job's resume to update its score.

- **The fixed template targets the saved Master CV.** Its Letter page size, margins,
  type sizes and blue headings are encoded in `backend/app/templates/finalized_cv.html`.
  The original source PDF is needed to verify an exact visual match. Other uploaded
  designs require a template change; detected layout metadata does not change CSS.
- **Fonts:** Windows uses installed Calibri; Docker installs the metrically compatible
  Carlito; the requested template stack falls back to Segoe UI/Arial when Calibri is absent. Exact glyph appearance requires the same fonts on each host.
- **One-page enforcement:** Content is never shrunk or clipped. Overlong CV downloads
  return HTTP 422; shorten the source wording or tailor again. Existing cached text is
  adapted to the template without a database migration.
- **Browser setup:** Local installations require `python -m playwright install chromium`;
  Linux hosts also need `python -m playwright install --with-deps chromium`. Docker
  installs Chromium and its dependencies during the build.
- **Structured output:** The model supplies `summary`, `core_skills`, and
  `experience_bullets`; the application supplies immutable contact details, category
  labels, employer headings, education, and additional sections. Jinja2 escapes all
  text and allows only application-generated bold spans.
- **Scraping:** Availability depends on the source sites and their rate limits.
