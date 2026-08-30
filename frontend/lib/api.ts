import type { Job, MasterCv, ScrapeResult, TailorResult } from "@/types/job";

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
  getMasterCv: () => request<MasterCv | null>("/api/cv"),

  listJobs: () => request<Job[]>("/api/jobs"),

  scrapeJobs: (primary_role: string, location: string) =>
    request<ScrapeResult>("/api/jobs/scrape", {
      method: "POST",
      body: JSON.stringify({ primary_role, location }),
    }),

  tailorJob: (id: number) => request<TailorResult>(`/api/jobs/${id}/tailor`, { method: "POST" }),

  toggleApplied: (id: number) => request<Job>(`/api/jobs/${id}/toggle-applied`, { method: "PATCH" }),

  downloadCv: async (id: number): Promise<{ blob: Blob; filename: string }> => {
    const res = await fetch(`${API_BASE_URL}/api/jobs/${id}/download-cv`);
    if (!res.ok) {
      const body = await res.text();
      throw new Error(`${res.status} ${res.statusText}: ${body}`);
    }
    const disposition = res.headers.get("Content-Disposition") ?? "";
    const match = disposition.match(/filename="?([^"]+)"?/);
    const filename = match?.[1] ?? `CV_${id}.pdf`;
    const blob = await res.blob();
    return { blob, filename };
  },
};
