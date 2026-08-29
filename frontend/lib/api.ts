import type { Job, Metrics, SmashResponse } from "@/types/job";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
    cache: "no-store",
  });
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${body}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  listJobs: (status?: string) =>
    request<Job[]>(`/api/jobs${status ? `?status=${status}` : ""}`),
  getMetrics: () => request<Metrics>("/api/jobs/metrics"),
  passJob: (id: number) => request<Job>(`/api/jobs/${id}/pass`, { method: "POST" }),
  smashJob: (id: number) => request<SmashResponse>(`/api/jobs/${id}/smash`, { method: "POST" }),
  ingestGmail: () => request<{ fetched: number; created: number; skipped: number }>(
    "/api/ingest/gmail",
    { method: "POST" }
  ),
  revalidate: () => request<{ checked: number; expired: number }>(
    "/api/ingest/revalidate",
    { method: "POST" }
  ),
  getCv: () => request<{ id: number; raw_text: string } | null>("/api/cv"),
  uploadCv: (raw_text: string) =>
    request<{ id: number; raw_text: string }>("/api/cv", {
      method: "POST",
      body: JSON.stringify({ raw_text }),
    }),
  uploadCvFile: async (file: File) => {
    const formData = new FormData();
    formData.append("file", file);
    const res = await fetch(`${API_BASE_URL}/api/cv/upload`, {
      method: "POST",
      body: formData,
    });
    if (!res.ok) {
      const body = await res.text();
      throw new Error(`${res.status} ${res.statusText}: ${body}`);
    }
    return res.json() as Promise<{ id: number; raw_text: string }>;
  },
};
