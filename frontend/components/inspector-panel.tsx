"use client";

import { Download, ExternalLink, Inbox, Loader2, Wand2 } from "lucide-react";

import { Pill } from "@/components/ui/pill";
import { cn } from "@/lib/utils";
import type { Job } from "@/types/job";

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

interface InspectorPanelProps {
  job: Job | null;
  tailoringId: number | null;
  downloadingId: number | null;
  onTailor: (job: Job) => void;
  onDownload: (job: Job) => void;
}

export function InspectorPanel({ job, tailoringId, downloadingId, onTailor, onDownload }: InspectorPanelProps) {
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
          <span className="text-[11px] font-medium uppercase tracking-wide text-zinc-500">Match score</span>
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
            disabled={isDownloading}
            className="flex h-8 flex-1 items-center justify-center gap-1.5 rounded-md bg-indigo-500 text-xs font-medium text-white transition-colors hover:bg-indigo-400 disabled:opacity-50"
          >
            {isDownloading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Download className="h-3.5 w-3.5" />}
            {isDownloading ? "Downloading…" : "Download PDF"}
          </button>
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4">
        <p className="mb-2 text-[11px] font-medium uppercase tracking-wide text-zinc-500">Job description</p>
        <p className="whitespace-pre-wrap text-xs leading-relaxed text-zinc-400">{job.description || "—"}</p>
      </div>
    </div>
  );
}
