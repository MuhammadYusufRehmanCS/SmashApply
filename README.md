# SmashApply

Local Cloud/DevOps job-application engine: live-scrapes job boards for a role plus 10 aligned
titles, tailors your Master CV's wording to each posting with a local LLM (Ollama), and generates
an ATS-friendly PDF that mirrors your Master CV's original layout — no paid cloud APIs, no managed
databases.

## Stack

- **Backend:** Python / FastAPI / SQLAlchemy / SQLite
- **Layout parsing:** `pypdf` (text) + `pdfplumber` (font/position metadata)
- **Scraping:** `python-jobspy` against LinkedIn, Indeed, Glassdoor, ZipRecruiter
- **Tailoring:** local Ollama endpoint (`llama3` by default)
- **PDF generation:** `reportlab`, driven by the parsed layout profile
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
│   │       ├── cv_tailor.py       # Ollama-backed keyword extraction + rewrite
│   │       └── pdf_generator.py   # reportlab ATS PDF mirroring the layout profile
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
cp .env.example .env
uvicorn app.main:app --reload
```

API serves on `http://127.0.0.1:8000` (docs at `/docs`). SQLite data lives in
`backend/data/smashapply.db`, created automatically on first run.

**Ollama setup:** install [Ollama](https://ollama.com), run `ollama pull llama3`, and make sure
`ollama serve` is running before using Tailor/Download.

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
3. Click **Tailor & Download CV** on any row: Ollama extracts the job's keywords, rewrites your
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

| Method | Path                          | Purpose                                              |
|--------|--------------------------------|-------------------------------------------------------|
| POST   | `/api/cv/upload`              | Upload Master CV (.pdf); parses text + layout profile |
| GET    | `/api/cv`                     | Fetch current master CV + layout profile              |
| POST   | `/api/jobs/scrape`             | Scrape primary role + 10 aligned titles, dedupe, save  |
| GET    | `/api/jobs`                    | List scraped jobs                                      |
| POST   | `/api/jobs/{id}/tailor`        | Ollama: extract keywords, rewrite CV bullets           |
| GET    | `/api/jobs/{id}/download-cv`   | Generate + download the tailored ATS PDF               |

## Known limitations

- **Layout parsing is heuristic.** Font/margin/column detection approximates the PDF's design —
  it can't recover exact original intent, only mirror proportions closely enough for reportlab to
  reproduce.
- **Site scraping depends on jobspy and the target sites.** Glassdoor requires a real city/state
  (not "Remote") to resolve a location, and ZipRecruiter frequently blocks datacenter/non-residential
  IPs with a 403. LinkedIn and Indeed are generally the most reliable of the four.
- **PDF fonts are mapped to reportlab's base-14 fonts** (Helvetica/Times/Courier family) by name
  matching — no custom TTF embedding.
