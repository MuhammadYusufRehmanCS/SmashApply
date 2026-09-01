import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "Smash Apply",
  description: "A focused dashboard for job tracking and resume-ready applications.",
  icons: {
    icon: "/favicon.svg",
  },
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}
