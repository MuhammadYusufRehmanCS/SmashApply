import type { Job, MasterCv, ScrapeResult, TailorResult } from "@/types/job";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

export type EmailReferral = { job_title: string; sender: string; jd_text: string; duration: string };

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
  extractLatestEmail: async (): Promise<EmailReferral> => {
    const res = await fetch(`${API_BASE_URL}/api/inbox/extract-latest`, { method: "POST", cache: "no-store" });
    const body = await res.json();
    if (!res.ok) throw new Error(body.detail ?? "Could not extract the email referral.");
    return body;
  },

  tailorEmail: async (referral: EmailReferral): Promise<Blob> => {
    const res = await fetch(`${API_BASE_URL}/api/inbox/tailor`, {
      method: "POST", cache: "no-store", headers: { "Content-Type": "application/json" }, body: JSON.stringify(referral),
    });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail ?? body.error ?? "Could not process the email referral. Please retry.");
    }
    return res.blob();
  },

  getMasterCv: () => request<MasterCv | null>("/api/cv"),

  uploadCv: async (file: File): Promise<MasterCv> => {
    const formData = new FormData();
    formData.append("file", file);
    // Can't go through request() -- it forces a JSON Content-Type header,
    // which breaks the multipart/form-data boundary the browser sets for us.
    const res = await fetch(`${API_BASE_URL}/api/cv/upload`, { method: "POST", body: formData, cache: "no-store" });
    if (!res.ok) {
      const body = await res.text();
      throw new Error(`${res.status} ${res.statusText}: ${body}`);
    }
    return res.json() as Promise<MasterCv>;
  },

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
