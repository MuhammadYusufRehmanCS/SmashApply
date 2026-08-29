import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "SmashApply",
  description: "Live Cloud/DevOps job scraping with local Ollama CV tailoring.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}
