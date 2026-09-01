"use client";

import { CheckCircle2, Circle, Download, Loader2 } from "lucide-react";

import { Pill, matchScoreTone } from "@/components/ui/pill";
import { cn } from "@/lib/utils";
import type { Job } from "@/types/job";

const ROW_GRID = "grid grid-cols-[1.1fr_1.6fr_1fr_92px_36px_36px] items-center gap-3";

interface JobTableProps {
  jobs: Job[];
  loading: boolean;
  selectedJobId: number | null;
  onSelect: (job: Job) => void;
  togglingId: number | null;
  onToggleApplied: (job: Job) => void;
  actioningId: number | null;
  onQuickDownload: (job: Job) => void;
}

export function JobTable({
  jobs,
  loading,
  selectedJobId,
  onSelect,
  togglingId,
  onToggleApplied,
  actioningId,
  onQuickDownload,
}: JobTableProps) {
  if (loading) {
    return <p className="p-6 text-xs text-zinc-500">Loading jobs...</p>;
  }

  if (jobs.length === 0) {
    return (
      <div className="p-6 text-xs text-zinc-500">
        No jobs found. Search to fill the board.
      </div>
    );
  }

  return (
    <div className="min-w-[640px]">
      <div
        className={cn(
          ROW_GRID,
          "sticky top-0 z-10 border-b border-white/10 bg-[#090A0D]/95 px-3 py-2 text-[11px] font-semibold uppercase tracking-wide text-zinc-500 backdrop-blur"
        )}
      >
        <span>Company</span>
        <span>Role</span>
        <span>Location</span>
        <span>Match</span>
        <span className="text-center">Applied</span>
        <span />
      </div>

      <div>
        {jobs.map((job) => {
          const selected = job.id === selectedJobId;
          return (
            <div
              key={job.id}
              onClick={() => onSelect(job)}
              className={cn(
                ROW_GRID,
                "cursor-pointer border-b border-white/[0.04] border-l-2 px-3 py-2 text-sm transition-colors hover:bg-white/[0.035]",
                selected ? "border-l-emerald-400 bg-white/[0.055]" : "border-l-transparent"
              )}
            >
              <div className="min-w-0">
                <p className="truncate font-medium text-zinc-100">{job.company}</p>
                <p className="truncate text-xs text-zinc-600">{job.site}</p>
              </div>

              <p className="truncate text-zinc-300">{job.title}</p>

              <p className="truncate text-zinc-500">{job.location || "--"}</p>

              <div>
                {job.match_score !== null ? (
                  <Pill tone={matchScoreTone(job.match_score)}>{job.match_score}%</Pill>
                ) : (
                  <Pill tone="zinc">--</Pill>
                )}
              </div>

              <div className="flex justify-center">
                <button
                  type="button"
                  title={job.applied ? "Mark as not applied" : "Mark as applied"}
                  onClick={(e) => {
                    e.stopPropagation();
                    onToggleApplied(job);
                  }}
                  disabled={togglingId === job.id}
                  className="flex h-6 w-6 items-center justify-center rounded text-zinc-600 transition-colors hover:text-emerald-300 disabled:opacity-50"
                >
                  {togglingId === job.id ? (
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  ) : job.applied ? (
                    <CheckCircle2 className="h-4 w-4 text-emerald-400" />
                  ) : (
                    <Circle className="h-4 w-4" />
                  )}
                </button>
              </div>

              <div className="flex justify-center">
                <button
                  type="button"
                  title="Tailor & download CV"
                  onClick={(e) => {
                    e.stopPropagation();
                    onQuickDownload(job);
                  }}
                  disabled={actioningId === job.id}
                  className="flex h-6 w-6 items-center justify-center rounded text-zinc-600 transition-colors hover:text-emerald-300 disabled:opacity-50"
                >
                  {actioningId === job.id ? (
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  ) : (
                    <Download className="h-3.5 w-3.5" />
                  )}
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
