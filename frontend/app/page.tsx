"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { FileText, Loader2, Search } from "lucide-react";

import { InspectorPanel } from "@/components/inspector-panel";
import { JobTable } from "@/components/job-table";
import { api } from "@/lib/api";
import type { Job, MasterCv } from "@/types/job";

function saveBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

export default function DashboardPage() {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [masterCv, setMasterCv] = useState<MasterCv | null>(null);
  const [uploading, setUploading] = useState(false);

  const [primaryRole, setPrimaryRole] = useState("Cloud Engineer / DevOps Engineer");
  const [location, setLocation] = useState("Remote");
  const [scraping, setScraping] = useState(false);

  const [jobs, setJobs] = useState<Job[]>([]);
  const [loadingJobs, setLoadingJobs] = useState(true);
  const [selectedJobId, setSelectedJobId] = useState<number | null>(null);

  const [togglingId, setTogglingId] = useState<number | null>(null);
  const [actioningId, setActioningId] = useState<number | null>(null);
  const [tailoringId, setTailoringId] = useState<number | null>(null);
  const [downloadingId, setDownloadingId] = useState<number | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  const totalApplied = useMemo(() => jobs.filter((job) => job.applied).length, [jobs]);
  const averageMatchScore = useMemo(() => {
    const scored = jobs.filter((job) => job.match_score !== null) as (Job & { match_score: number })[];
    if (scored.length === 0) return null;
    return Math.round(scored.reduce((sum, job) => sum + job.match_score, 0) / scored.length);
  }, [jobs]);

  const selectedJob = useMemo(() => jobs.find((job) => job.id === selectedJobId) ?? null, [jobs, selectedJobId]);

  const refreshJobs = useCallback(async () => {
    const data = await api.listJobs();
    setJobs(data);
  }, []);

  useEffect(() => {
    api
      .getMasterCv()
      .then(setMasterCv)
      .catch(() => setMasterCv(null));

    api
      .listJobs()
      .then(setJobs)
      .catch(() => setJobs([]))
      .finally(() => setLoadingJobs(false));
  }, []);

  // Default to the first job once the list loads, so the inspector isn't empty.
  useEffect(() => {
    if (!loadingJobs && selectedJobId === null && jobs.length > 0) {
      setSelectedJobId(jobs[0].id);
    }
  }, [loadingJobs, jobs, selectedJobId]);

  async function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;

    setUploading(true);
    setActionError(null);
    try {
      const cv = await api.uploadCv(file);
      setMasterCv(cv);
    } catch (err) {
      setActionError(err instanceof Error ? err.message : String(err));
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  async function handleScrape() {
    setScraping(true);
    setActionError(null);
    try {
      await api.scrapeJobs(primaryRole, location);
      await refreshJobs();
    } catch (err) {
      setActionError(err instanceof Error ? err.message : String(err));
    } finally {
      setScraping(false);
    }
  }

  async function handleTailorAndDownload(job: Job) {
    setActioningId(job.id);
    setActionError(null);
    try {
      await api.tailorJob(job.id);
      const { blob, filename } = await api.downloadCv(job.id);
      saveBlob(blob, filename);
      await refreshJobs();
    } catch (err) {
      setActionError(err instanceof Error ? err.message : String(err));
    } finally {
      setActioningId(null);
    }
  }

  async function handleTailor(job: Job) {
    setTailoringId(job.id);
    setActionError(null);
    try {
      await api.tailorJob(job.id);
      await refreshJobs();
    } catch (err) {
      setActionError(err instanceof Error ? err.message : String(err));
    } finally {
      setTailoringId(null);
    }
  }

  async function handleDownload(job: Job) {
    setDownloadingId(job.id);
    setActionError(null);
    try {
      const { blob, filename } = await api.downloadCv(job.id);
      saveBlob(blob, filename);
      await refreshJobs();
    } catch (err) {
      setActionError(err instanceof Error ? err.message : String(err));
    } finally {
      setDownloadingId(null);
    }
  }

  async function handleToggleApplied(job: Job) {
    setTogglingId(job.id);
    setActionError(null);
    try {
      const updated = await api.toggleApplied(job.id);
      setJobs((prev) => prev.map((j) => (j.id === updated.id ? updated : j)));
    } catch (err) {
      setActionError(err instanceof Error ? err.message : String(err));
    } finally {
      setTogglingId(null);
    }
  }

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-[#08090B] text-zinc-100">
      <header className="shrink-0 border-b border-white/10 bg-[#0C0D10]/95 px-4 shadow-[0_1px_0_rgba(255,255,255,0.03)]">
        <div className="flex min-h-16 items-center gap-4">
          <div className="flex min-w-[184px] items-center gap-3">
            <span
              aria-hidden="true"
              className="relative flex h-9 w-9 shrink-0 items-center justify-center overflow-hidden rounded-md border border-emerald-400/35 bg-[#101417] shadow-[inset_0_1px_0_rgba(255,255,255,0.08)]"
            >
              <span className="absolute -left-2 top-3 h-[2px] w-14 -rotate-12 bg-emerald-400/80" />
              <span className="relative text-[13px] font-black tracking-tight text-white">SA</span>
            </span>
            <div className="min-w-0">
              <p className="text-[15px] font-semibold leading-none tracking-tight">
                <span className="text-zinc-100">Smash</span>
                <span className="text-emerald-300"> Apply</span>
              </p>
              <p className="mt-1 text-[10px] font-medium uppercase tracking-[0.18em] text-zinc-500">
                Job search cockpit
              </p>
            </div>
          </div>

          <div className="hidden h-8 w-px bg-white/10 md:block" />

          <div className="flex min-w-0 flex-1 items-center gap-2">
            <label htmlFor="primary-role" className="sr-only">
              Job title
            </label>
            <input
              id="primary-role"
              value={primaryRole}
              onChange={(e) => setPrimaryRole(e.target.value)}
              placeholder="Job title"
              className="h-9 w-64 max-w-[36vw] rounded-md border border-white/10 bg-[#111318] px-3 text-xs font-medium text-zinc-200 placeholder:text-zinc-600 outline-none transition focus:border-emerald-400/50 focus:ring-2 focus:ring-emerald-400/10"
            />
            <label htmlFor="job-location" className="sr-only">
              Location
            </label>
            <input
              id="job-location"
              value={location}
              onChange={(e) => setLocation(e.target.value)}
              placeholder="Location"
              className="h-9 w-36 rounded-md border border-white/10 bg-[#111318] px-3 text-xs font-medium text-zinc-200 placeholder:text-zinc-600 outline-none transition focus:border-emerald-400/50 focus:ring-2 focus:ring-emerald-400/10"
            />
            <button
              type="button"
              onClick={handleScrape}
              disabled={scraping}
              className="flex h-9 items-center gap-1.5 whitespace-nowrap rounded-md bg-emerald-400 px-3 text-xs font-semibold text-zinc-950 shadow-[0_0_0_1px_rgba(255,255,255,0.12)_inset] transition-colors hover:bg-emerald-300 disabled:opacity-50"
            >
              {scraping ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Search className="h-3.5 w-3.5" />}
              Find Jobs
            </button>
          </div>

          <div className="ml-auto flex items-center gap-2">
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              disabled={uploading}
              title={masterCv ? masterCv.filename : "Upload Master CV"}
              className="flex h-9 items-center gap-1.5 rounded-md border border-white/10 bg-[#111318] px-2.5 text-xs font-medium text-zinc-300 transition-colors hover:border-emerald-400/35 hover:text-zinc-100 disabled:opacity-50"
            >
              {uploading ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <FileText className="h-3.5 w-3.5" />
              )}
              <span className="max-w-[140px] truncate">{masterCv ? masterCv.filename : "Upload CV"}</span>
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf"
              onChange={handleFileChange}
              className="hidden"
            />

            <div className="hidden items-center gap-2 rounded-md border border-white/10 bg-[#111318] px-2.5 py-1.5 font-mono text-[11px] text-zinc-500 lg:flex">
              <span>
                <span className="text-zinc-200">{totalApplied}</span> applied
              </span>
              <span className="h-3 w-px bg-white/10" />
              <span>
                <span className="text-emerald-300">
                  {averageMatchScore !== null ? `${averageMatchScore}%` : "--"}
                </span>{" "}
                avg match
              </span>
            </div>
          </div>
        </div>
      </header>

      {actionError && (
        <div className="shrink-0 border-b border-red-500/20 bg-red-500/10 px-4 py-1.5 text-xs text-red-400">
          {actionError}
        </div>
      )}

      <div className="flex min-h-0 flex-1 border-t border-white/[0.03]">
        <section className="w-[65%] min-w-0 overflow-y-auto border-r border-white/10 bg-[#090A0D]">
          <JobTable
            jobs={jobs}
            loading={loadingJobs}
            selectedJobId={selectedJobId}
            onSelect={(job) => setSelectedJobId(job.id)}
            togglingId={togglingId}
            onToggleApplied={handleToggleApplied}
            actioningId={actioningId}
            onQuickDownload={handleTailorAndDownload}
          />
        </section>

        <aside className="flex w-[35%] min-w-0 flex-col overflow-hidden bg-[#0B0C0F]">
          <InspectorPanel
            job={selectedJob}
            tailoringId={tailoringId}
            downloadingId={downloadingId}
            onTailor={handleTailor}
            onDownload={handleDownload}
          />
        </aside>
      </div>
    </div>
  );
}
