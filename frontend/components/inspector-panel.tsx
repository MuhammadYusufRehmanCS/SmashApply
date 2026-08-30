"use client";

import { useEffect, useState } from "react";
import { CheckCircle2, Download, ExternalLink, Inbox, Loader2, Wand2 } from "lucide-react";

import { Pill } from "@/components/ui/pill";
import { cn } from "@/lib/utils";
import type { Job } from "@/types/job";

type Tab = "description" | "tailored";

function gaugeBarColor(score: number): string {
  if (score >= 75) return "bg-emerald-500";
  if (score >= 50) return "bg-amber-500";
  return "bg-red-500";
}

function gaugeTextColor(score: number): string {
  if (score >= 75) return "text-emerald-400";
  if (score >= 50) return "text-amber-400";
  return "text-red-400";
}

// The backend normalizes every bullet to a leading "-" for newly-tailored
// jobs, but that fix can't reach CVs already tailored and stored before it
// shipped -- so this also recognizes the other glyphs the model sometimes
// used instead ("•" etc., plus the "�" fallback for Ollama's occasional
// mangled-encoding bullet artifact) as a bullet marker.
const BULLET_PREFIX_RE = /^[-•◦▪●‣·�]+\s*/;

// The tailoring prompt wraps 2-4 scannable keywords per bullet in **double
// asterisks** (the PDF generator renders those spans bold too), so a bullet
// carrying "**" is exactly the model's own signal that this line is one of
// the "key" tailored highlights, not just a reproduced original bullet.
function extractHighlightBullets(tailoredCv: string, limit = 6): string[] {
  const bulletLines = tailoredCv
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => BULLET_PREFIX_RE.test(line));

  // Prefer bullets the model actually bolded (its own signal that these are
  // the tailored-for-this-job highlights) -- but smaller local models don't
  // always apply that formatting, so fall back to the first few bullets
  // rather than showing an empty, seemingly-broken section.
  const bolded = bulletLines.filter((line) => line.includes("**"));
  const source = bolded.length > 0 ? bolded : bulletLines;

  return source.map((line) => line.replace(BULLET_PREFIX_RE, "")).slice(0, limit);
}

function renderBolded(text: string) {
  return text.split(/(\*\*[^*]+\*\*)/g).map((part, i) =>
    part.startsWith("**") && part.endsWith("**") ? (
      <strong key={i} className="font-semibold text-indigo-300">
        {part.slice(2, -2)}
      </strong>
    ) : (
      <span key={i}>{part}</span>
    )
  );
}

interface InspectorPanelProps {
  job: Job | null;
  tailoringId: number | null;
  downloadingId: number | null;
  onTailor: (job: Job) => void;
  onDownload: (job: Job) => void;
}

export function InspectorPanel({ job, tailoringId, downloadingId, onTailor, onDownload }: InspectorPanelProps) {
  const [activeTab, setActiveTab] = useState<Tab>("description");

  // Switching jobs while the "Tailored CV Preview" tab is open would otherwise
  // silently show the newly-selected job's stale/absent tailoring state.
  useEffect(() => {
    setActiveTab("description");
  }, [job?.id]);

  if (!job) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-2 px-6 text-center">
        <Inbox className="h-5 w-5 text-zinc-700" />
        <p className="text-xs text-zinc-600">Select a job to inspect</p>
      </div>
    );
  }

  const isTailoring = tailoringId === job.id;
  const isDownloading = downloadingId === job.id;
  const score = job.match_score;
  const isTailored = job.tailored_at !== null;

  return (
    <div className="flex h-full flex-col">
      <div className="shrink-0 border-b border-zinc-800 px-4 py-4">
        <div className="flex items-start justify-between gap-2">
          <h2 className="text-sm font-semibold leading-snug text-zinc-100">{job.title}</h2>
          <a
            href={job.job_url}
            target="_blank"
            rel="noopener noreferrer"
            title="Open original posting"
            className="mt-0.5 shrink-0 text-zinc-600 hover:text-zinc-300"
          >
            <ExternalLink className="h-3.5 w-3.5" />
          </a>
        </div>
        <p className="mt-1 truncate text-xs text-zinc-500">
          {job.company} · {job.location || "—"}
        </p>
        <div className="mt-2 flex flex-wrap gap-1.5">
          <Pill tone="indigo">{job.role_category}</Pill>
          <Pill tone="zinc">{job.site}</Pill>
        </div>
      </div>

      <div className="shrink-0 border-b border-zinc-800 px-4 py-4">
        <div className="mb-1.5 flex items-center justify-between">
          <div className="flex items-center gap-1.5">
            <span className="text-[11px] font-medium uppercase tracking-wide text-zinc-500">Match score</span>
            {isTailored && (
              <span title="Tailored successfully" className="flex items-center text-emerald-400">
                <CheckCircle2 className="h-3 w-3" />
              </span>
            )}
          </div>
          {score !== null ? (
            <span className={cn("font-mono text-sm font-semibold", gaugeTextColor(score))}>{score}%</span>
          ) : (
            <span className="font-mono text-sm text-zinc-600">—</span>
          )}
        </div>
        <div className="h-1.5 w-full overflow-hidden rounded-full bg-zinc-800">
          {score !== null && (
            <div
              className={cn("h-full rounded-full transition-all", gaugeBarColor(score))}
              style={{ width: `${score}%` }}
            />
          )}
        </div>
        {score === null && <p className="mt-1.5 text-[11px] text-zinc-600">Not tailored yet</p>}

        <div className="mt-3 flex gap-2">
          <button
            type="button"
            onClick={() => onTailor(job)}
            disabled={isTailoring}
            className="flex h-8 flex-1 items-center justify-center gap-1.5 rounded-md border border-zinc-800 bg-zinc-900 text-xs font-medium text-zinc-200 transition-colors hover:border-zinc-700 disabled:opacity-50"
          >
            {isTailoring ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Wand2 className="h-3.5 w-3.5" />}
            {isTailoring ? "Tailoring…" : "Tailor CV"}
          </button>
          <button
            type="button"
            onClick={() => onDownload(job)}
            disabled={isDownloading || !isTailored}
            title={isTailored ? undefined : "Tailor the CV first"}
            className="flex h-8 flex-1 items-center justify-center gap-1.5 rounded-md bg-indigo-500 text-xs font-medium text-white transition-colors hover:bg-indigo-400 disabled:cursor-not-allowed disabled:bg-zinc-800 disabled:text-zinc-500"
          >
            {isDownloading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Download className="h-3.5 w-3.5" />}
            {isDownloading ? "Downloading…" : "Download PDF"}
          </button>
        </div>
        {!isTailored && <p className="mt-1.5 text-[11px] text-zinc-600">Tailor the CV to enable PDF download.</p>}
      </div>

      <div className="flex shrink-0 gap-4 border-b border-zinc-800 px-4">
        <button
          type="button"
          onClick={() => setActiveTab("description")}
          className={cn(
            "border-b-2 py-2.5 text-xs font-medium transition-colors",
            activeTab === "description"
              ? "border-indigo-500 text-zinc-100"
              : "border-transparent text-zinc-500 hover:text-zinc-300"
          )}
        >
          Job Description
        </button>
        <button
          type="button"
          onClick={() => setActiveTab("tailored")}
          className={cn(
            "border-b-2 py-2.5 text-xs font-medium transition-colors",
            activeTab === "tailored"
              ? "border-indigo-500 text-zinc-100"
              : "border-transparent text-zinc-500 hover:text-zinc-300"
          )}
        >
          Tailored CV Preview
        </button>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4">
        {activeTab === "description" ? (
          <p className="whitespace-pre-wrap text-xs leading-relaxed text-zinc-400">{job.description || "—"}</p>
        ) : job.tailored_cv ? (
          <div className="flex flex-col gap-4">
            <div>
              <p className="mb-1.5 text-[11px] font-medium uppercase tracking-wide text-zinc-500">Status</p>
              <p className="flex items-center gap-1.5 text-xs text-emerald-400">
                <CheckCircle2 className="h-3.5 w-3.5" />
                Tailored{" "}
                {job.tailored_at &&
                  new Date(job.tailored_at).toLocaleString(undefined, {
                    month: "short",
                    day: "numeric",
                    hour: "numeric",
                    minute: "2-digit",
                  })}
              </p>
            </div>

            {job.tailored_keywords && (
              <div>
                <p className="mb-1.5 text-[11px] font-medium uppercase tracking-wide text-zinc-500">
                  Matched skills
                </p>
                <div className="flex flex-wrap gap-1.5">
                  {job.tailored_keywords
                    .split(",")
                    .map((kw) => kw.trim())
                    .filter(Boolean)
                    .map((kw) => (
                      <Pill key={kw} tone="zinc">
                        {kw}
                      </Pill>
                    ))}
                </div>
              </div>
            )}

            <div>
              <p className="mb-1.5 text-[11px] font-medium uppercase tracking-wide text-zinc-500">
                Tailored bullet highlights
              </p>
              <ul className="flex flex-col gap-2">
                {extractHighlightBullets(job.tailored_cv).map((bullet, i) => (
                  <li key={i} className="text-xs leading-relaxed text-zinc-400">
                    <span className="mr-1.5 text-zinc-600">–</span>
                    {renderBolded(bullet)}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        ) : (
          <div className="flex h-full flex-col items-center justify-center gap-2 text-center">
            <Wand2 className="h-5 w-5 text-zinc-700" />
            <p className="text-xs text-zinc-600">
              Not tailored yet. Click &quot;Tailor CV&quot; above to generate matched bullets and skills.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
