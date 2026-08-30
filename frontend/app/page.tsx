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
    <div className="flex h-screen flex-col bg-[#0D0E11] text-zinc-100">
      <header className="flex h-14 shrink-0 items-center gap-3 border-b border-zinc-800 px-4">
        <span className="font-semibold tracking-tight text-zinc-100">SmashApply</span>
        <div className="h-5 w-px bg-zinc-800" />

        <input
          value={primaryRole}
          onChange={(e) => setPrimaryRole(e.target.value)}
          placeholder="Job title"
          className="h-8 w-56 rounded-md border border-zinc-800 bg-zinc-900 px-2.5 text-xs text-zinc-200 placeholder:text-zinc-600 focus:outline-none focus:ring-1 focus:ring-indigo-500"
        />
        <input
          value={location}
          onChange={(e) => setLocation(e.target.value)}
          placeholder="Location"
          className="h-8 w-32 rounded-md border border-zinc-800 bg-zinc-900 px-2.5 text-xs text-zinc-200 placeholder:text-zinc-600 focus:outline-none focus:ring-1 focus:ring-indigo-500"
        />
        <button
          type="button"
          onClick={handleScrape}
          disabled={scraping}
          className="flex h-8 items-center gap-1.5 whitespace-nowrap rounded-md bg-indigo-500 px-3 text-xs font-medium text-white transition-colors hover:bg-indigo-400 disabled:opacity-50"
        >
          {scraping ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Search className="h-3.5 w-3.5" />}
          Scrape
        </button>

        <div className="ml-auto flex items-center gap-3">
          <button
            type="button"
            onClick={() => fileInputRef.current?.click()}
            disabled={uploading}
            title={masterCv ? masterCv.filename : "Upload Master CV"}
            className="flex h-8 items-center gap-1.5 rounded-md border border-zinc-800 bg-zinc-900 px-2.5 text-xs text-zinc-300 transition-colors hover:border-zinc-700 disabled:opacity-50"
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

          <span className="font-mono text-xs text-zinc-500">
            {totalApplied} Applied • {averageMatchScore !== null ? `${averageMatchScore}%` : "—"} Avg Match
          </span>
        </div>
      </header>

      {actionError && (
        <div className="shrink-0 border-b border-red-500/20 bg-red-500/10 px-4 py-1.5 text-xs text-red-400">
          {actionError}
        </div>
      )}

      <div className="flex min-h-0 flex-1">
        <section className="w-[65%] min-w-0 overflow-y-auto border-r border-zinc-800">
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

        <aside className="flex w-[35%] min-w-0 flex-col overflow-hidden">
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
