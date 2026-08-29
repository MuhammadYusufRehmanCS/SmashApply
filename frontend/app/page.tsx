"use client";

import { useCallback, useEffect, useState } from "react";
import { FileText, Inbox, RefreshCw } from "lucide-react";

import { CvManager } from "@/components/cv-manager";
import { CvPreview } from "@/components/cv-preview";
import { JobCard } from "@/components/job-card";
import { MetricsBar } from "@/components/metrics-bar";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";
import type { Job, Metrics } from "@/types/job";

export default function DashboardPage() {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [metrics, setMetrics] = useState<Metrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [syncing, setSyncing] = useState(false);
  const [smashingId, setSmashingId] = useState<number | null>(null);
  const [showCvManager, setShowCvManager] = useState(false);
  const [preview, setPreview] = useState<{ title: string; text: string } | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    const [jobsData, metricsData] = await Promise.all([
      api.listJobs("unseen,smashing,expired"),
      api.getMetrics(),
    ]);
    setJobs(jobsData);
    setMetrics(metricsData);
  }, []);

  useEffect(() => {
    refresh()
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false));
  }, [refresh]);

  async function handlePass(id: number) {
    setJobs((prev) => prev.filter((j) => j.id !== id));
    try {
      await api.passJob(id);
      const metricsData = await api.getMetrics();
      setMetrics(metricsData);
    } catch (e) {
      setError(String(e));
      refresh();
    }
  }

  async function handleSmash(id: number) {
    setSmashingId(id);
    setError(null);
    try {
      const result = await api.smashJob(id);
      setJobs((prev) => prev.filter((j) => j.id !== id));
      const metricsData = await api.getMetrics();
      setMetrics(metricsData);
      setPreview({ title: result.job.title, text: result.tailored_cv });
      window.open(result.apply_url, "_blank", "noopener,noreferrer");
    } catch (e) {
      setError(String(e));
    } finally {
      setSmashingId(null);
    }
  }

  async function handleSync() {
    setSyncing(true);
    setError(null);
    try {
      await api.ingestGmail();
      await refresh();
    } catch (e) {
      setError(String(e));
    } finally {
      setSyncing(false);
    }
  }

  return (
    <main className="mx-auto max-w-6xl px-4 py-8">
      <div className="mb-6 flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-3xl font-bold">SmashApply</h1>
          <p className="text-sm text-muted-foreground">Smash or Pass your job pipeline.</p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" onClick={() => setShowCvManager(true)}>
            <FileText className="h-4 w-4" />
            Master CV
          </Button>
          <Button variant="outline" onClick={handleSync} disabled={syncing}>
            <Inbox className="h-4 w-4" />
            {syncing ? "Syncing..." : "Sync Gmail"}
          </Button>
          <Button variant="outline" onClick={refresh}>
            <RefreshCw className="h-4 w-4" />
          </Button>
        </div>
      </div>

      <div className="mb-8">
        <MetricsBar metrics={metrics} />
      </div>

      {error && (
        <div className="mb-6 rounded-md border border-pass bg-pass/10 p-3 text-sm text-pass">
          {error}
        </div>
      )}

      {loading ? (
        <p className="text-muted-foreground">Loading jobs...</p>
      ) : jobs.length === 0 ? (
        <div className="rounded-lg border border-dashed border-border p-12 text-center text-muted-foreground">
          No jobs in the pipeline yet. Click <span className="font-medium">Sync Gmail</span> to
          pull in new alerts.
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {jobs.map((job) => (
            <JobCard
              key={job.id}
              job={job}
              onPass={handlePass}
              onSmash={handleSmash}
              isSmashing={smashingId === job.id}
            />
          ))}
        </div>
      )}

      {showCvManager && <CvManager onClose={() => setShowCvManager(false)} />}
      {preview && (
        <CvPreview
          tailoredCv={preview.text}
          jobTitle={preview.title}
          onClose={() => setPreview(null)}
        />
      )}
    </main>
  );
}
