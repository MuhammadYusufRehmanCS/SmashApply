export interface Job {
  id: number;
  title: string;
  company: string;
  location: string;
  job_url: string;
  site: string;
  role_category: string;
  is_primary_role: boolean;
  description: string;
  date_posted: string | null;
  tailored_keywords: string | null;
  tailored_at: string | null;
  tailored_cv: string | null;
  match_score: number | null;
  applied: boolean;
  created_at: string;
}

export interface ScrapeResult {
  primary_role: string;
  location: string;
  roles_queried: string[];
  total_found: number;
  created: number;
  skipped: number;
  site_errors: string[];
}

export interface MasterCvSection {
  name: string;
  content: string;
}

export interface MasterCvLayout {
  font_family: string;
  body_font_size: number;
  heading_font_size: number;
  line_spacing_ratio: number;
  margins: { left: number; right: number; top: number; bottom: number };
  page_width: number;
  page_height: number;
  column_count: number;
  section_order: string[];
}

export interface MasterCv {
  id: number;
  filename: string;
  raw_text: string;
  sections: MasterCvSection[];
  layout: MasterCvLayout;
  created_at: string;
  updated_at: string;
}

export interface TailorResult {
  job_id: number;
  keywords: string[];
  tailored_cv: string;
}
