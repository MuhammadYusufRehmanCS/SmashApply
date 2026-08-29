export type JobStatus = "unseen" | "smashing" | "smashed" | "passed" | "expired";

export interface Job {
  id: number;
  title: string;
  company: string;
  raw_url: string;
  final_url: string | null;
  description: string;
  match_score: number;
  status: JobStatus;
  validation_note: string | null;
  created_at: string;
  updated_at: string;
}

export interface Metrics {
  smashed: number;
  passed: number;
  active_pipeline: number;
  total: number;
}

export interface SmashResponse {
  job: Job;
  tailored_cv: string;
  apply_url: string;
}
