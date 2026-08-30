"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { CheckCircle2, Circle, Download, FileText, Loader2, Search, Target, Upload } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
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

const JOBS_LOAD_TIMEOUT_MS = 5000;

function matchScoreBadgeClasses(score: number): string {
  if (score >= 75) return "border-transparent bg-smash text-smash-foreground";
  if (score >= 50) return "border-transparent bg-amber-500 text-white";
  return "border-transparent bg-pass text-pass-foreground";
}

export default function DashboardPage() {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [masterCv, setMasterCv] = useState<MasterCv | null>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const [primaryRole, setPrimaryRole] = useState("Cloud Engineer / DevOps Engineer");
  const [location, setLocation] = useState("Remote");
  const [scraping, setScraping] = useState(false);
  const [scrapeStatus, setScrapeStatus] = useState<string | null>(null);

  const [jobs, setJobs] = useState<Job[]>([]);
  const [loadingJobs, setLoadingJobs] = useState(true);
  const [actioningId, setActioningId] = useState<number | null>(null);
  const [togglingId, setTogglingId] = useState<number | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);

  const totalApplied = useMemo(() => jobs.filter((job) => job.applied).length, [jobs]);
  const averageMatchScore = useMemo(() => {
    const scored = jobs.filter((job) => job.match_score !== null) as (Job & { match_score: number })[];
    if (scored.length === 0) return null;
    return Math.round(scored.reduce((sum, job) => sum + job.match_score, 0) / scored.length);
  }, [jobs]);

  const refreshJobs = useCallback(async () => {
    const data = await api.listJobs();
    setJobs(data);
  }, []);

  useEffect(() => {
    api
      .getMasterCv()
      .then(setMasterCv)
      .catch(() => setMasterCv(null));

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), JOBS_LOAD_TIMEOUT_MS);

    fetch("http://127.0.0.1:8000/api/jobs", { signal: controller.signal })
      .then((res) => {
        if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
        return res.json() as Promise<Job[]>;
      })
      .then(setJobs)
      .catch(() => setJobs([]))
      .finally(() => {
        clearTimeout(timeoutId);
        setLoadingJobs(false);
      });

    return () => {
      clearTimeout(timeoutId);
      controller.abort();
    };
  }, []);

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    setSelectedFile(e.target.files?.[0] ?? null);
    setUploadError(null);
  }

  async function handleUpload() {
    if (!selectedFile) return;

    setUploading(true);
    setUploadError(null);
    try {
      const formData = new FormData();
      formData.append("file", selectedFile);
      const res = await fetch("http://127.0.0.1:8000/api/cv/upload", {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        const body = await res.text();
        throw new Error(`${res.status} ${res.statusText}${body ? `: ${body}` : ""}`);
      }
      const cv: MasterCv = await res.json();
      setMasterCv(cv);
      setSelectedFile(null);
      if (fileInputRef.current) fileInputRef.current.value = "";
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : String(err));
    } finally {
      setUploading(false);
    }
  }

  async function handleScrape() {
    setScraping(true);
    setScrapeStatus(null);
    setActionError(null);
    try {
      const result = await api.scrapeJobs(primaryRole, location);
      setScrapeStatus(
        `Found ${result.total_found} across ${result.roles_queried.length} roles — added ${result.created}, skipped ${result.skipped} duplicates.` +
          (result.site_errors.length ? ` (${result.site_errors.length} role queries failed)` : "")
      );
      await refreshJobs();
    } catch (err) {
      setActionError(String(err));
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
      setActionError(String(err));
    } finally {
      setActioningId(null);
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
    <main className="mx-auto max-w-6xl px-4 py-8">
      {/* Top bar */}
      <div className="mb-6 flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold">SmashApply</h1>
          <p className="text-sm text-muted-foreground">
            Live Cloud/DevOps job scraping, local Ollama CV tailoring, ATS PDF generation.
          </p>
        </div>

        <Card className="w-full max-w-md">
          <CardContent className="flex flex-col gap-2 p-4">
            <div className="flex items-center gap-2 text-sm font-medium">
              <FileText className="h-4 w-4" />
              Master CV
            </div>

            <div className="flex items-center gap-2">
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf"
                onChange={handleFileChange}
                className="min-w-0 flex-1 text-xs text-muted-foreground file:mr-2 file:rounded-md file:border-0 file:bg-blue-600 file:px-2 file:py-1 file:text-xs file:text-white file:hover:bg-blue-500"
              />
              <Button size="sm" variant="outline" onClick={handleUpload} disabled={uploading || !selectedFile}>
                {uploading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Upload className="h-4 w-4" />}
                {uploading ? "Uploading..." : "Upload PDF"}
              </Button>
            </div>

            {uploadError && <p className="text-xs text-pass">Upload failed: {uploadError}</p>}

            <p className="text-xs font-medium">
              {masterCv ? `Master CV Uploaded: ${masterCv.filename}` : "No Master CV uploaded yet."}
            </p>

            {masterCv && (
              <div className="rounded-md border border-border bg-muted/40 p-2 text-xs text-muted-foreground">
                <p>Sections: {masterCv.layout.section_order.join(" → ")}</p>
                <p>
                  Font: {masterCv.layout.font_family} · Body {masterCv.layout.body_font_size}pt / Heading{" "}
                  {masterCv.layout.heading_font_size}pt · {masterCv.layout.column_count}-column
                </p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Search control */}
      <Card className="mb-6">
        <CardHeader>
          <CardTitle className="text-base">Scrape Cloud/DevOps Jobs</CardTitle>
        </CardHeader>
        <CardContent className="flex flex-wrap items-end gap-3 pt-0">
          <div className="flex min-w-[240px] flex-1 flex-col gap-1">
            <label className="text-xs font-medium text-muted-foreground">Job Title</label>
            <input
              className="h-10 rounded-md border border-border bg-transparent px-3 text-sm"
              value={primaryRole}
              onChange={(e) => setPrimaryRole(e.target.value)}
            />
          </div>
          <div className="flex min-w-[160px] flex-col gap-1">
            <label className="text-xs font-medium text-muted-foreground">Location</label>
            <input
              className="h-10 rounded-md border border-border bg-transparent px-3 text-sm"
              value={location}
              onChange={(e) => setLocation(e.target.value)}
            />
          </div>
          <Button onClick={handleScrape} disabled={scraping}>
            {scraping ? <Loader2 className="h-4 w-4 animate-spin" /> : <Search className="h-4 w-4" />}
            {scraping ? "Scraping..." : "Scrape Jobs"}
          </Button>
        </CardContent>
        {scrapeStatus && <p className="px-4 pb-4 text-xs text-muted-foreground">{scrapeStatus}</p>}
      </Card>

      {/* Stat bar */}
      <div className="mb-6 grid grid-cols-1 gap-4 sm:max-w-md sm:grid-cols-2">
        <Card>
          <CardContent className="flex items-center gap-3 p-4">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-smash/15 text-smash">
              <CheckCircle2 className="h-5 w-5" />
            </div>
            <div>
              <p className="text-xs text-muted-foreground">Total applied</p>
              <p className="text-2xl font-semibold">{totalApplied}</p>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="flex items-center gap-3 p-4">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-primary/15 text-primary">
              <Target className="h-5 w-5" />
            </div>
            <div>
              <p className="text-xs text-muted-foreground">Average tailored score</p>
              <p className="text-2xl font-semibold">
                {averageMatchScore !== null ? `${averageMatchScore}%` : "—"}
              </p>
            </div>
          </CardContent>
        </Card>
      </div>

      {actionError && (
        <div className="mb-4 rounded-md border border-pass bg-pass/10 p-3 text-sm text-pass">{actionError}</div>
      )}

      {/* Job table */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Jobs ({jobs.length})</CardTitle>
        </CardHeader>
        <CardContent className="pt-0">
          {loadingJobs ? (
            <p className="text-sm text-muted-foreground">Loading jobs...</p>
          ) : jobs.length === 0 ? (
            <div className="rounded-lg border border-dashed border-border p-10 text-center text-sm text-muted-foreground">
              No jobs found. Click <span className="font-medium">&apos;Scrape Jobs&apos;</span> to fetch live
              postings.
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead>
                  <tr className="border-b border-border text-xs uppercase text-muted-foreground">
                    <th className="py-2 pr-3">Title</th>
                    <th className="py-2 pr-3">Company</th>
                    <th className="py-2 pr-3">Aligned Role Category</th>
                    <th className="py-2 pr-3">Location</th>
                    <th className="py-2 pr-3">Source</th>
                    <th className="py-2 pr-3">Match Score</th>
                    <th className="py-2 pr-3">Applied</th>
                    <th className="py-2 pr-3" />
                  </tr>
                </thead>
                <tbody>
                  {jobs.map((job) => (
                    <tr key={job.id} className="border-b border-border/60 align-top">
                      <td className="py-3 pr-3 font-medium">
                        <a
                          href={job.job_url}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="hover:underline"
                        >
                          {job.title}
                        </a>
                      </td>
                      <td className="py-3 pr-3">{job.company}</td>
                      <td className="py-3 pr-3">
                        <Badge variant={job.is_primary_role ? "success" : "muted"}>{job.role_category}</Badge>
                      </td>
                      <td className="py-3 pr-3">{job.location}</td>
                      <td className="py-3 pr-3 text-xs text-muted-foreground">{job.site}</td>
                      <td className="py-3 pr-3">
                        {job.match_score !== null ? (
                          <Badge className={matchScoreBadgeClasses(job.match_score)}>
                            {job.match_score}% Match
                          </Badge>
                        ) : (
                          <span className="text-xs text-muted-foreground">Not tailored</span>
                        )}
                      </td>
                      <td className="py-3 pr-3">
                        <Button
                          size="sm"
                          variant={job.applied ? "smash" : "outline"}
                          onClick={() => handleToggleApplied(job)}
                          disabled={togglingId === job.id}
                        >
                          {togglingId === job.id ? (
                            <Loader2 className="h-4 w-4 animate-spin" />
                          ) : job.applied ? (
                            <CheckCircle2 className="h-4 w-4" />
                          ) : (
                            <Circle className="h-4 w-4" />
                          )}
                          {job.applied ? "Applied" : "Mark as Applied"}
                        </Button>
                      </td>
                      <td className="py-3 pr-3">
                        <Button
                          size="sm"
                          onClick={() => handleTailorAndDownload(job)}
                          disabled={actioningId === job.id}
                        >
                          {actioningId === job.id ? (
                            <Loader2 className="h-4 w-4 animate-spin" />
                          ) : (
                            <Download className="h-4 w-4" />
                          )}
                          {actioningId === job.id ? "Tailoring..." : "Tailor & Download CV"}
                        </Button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
    </main>
  );
}
