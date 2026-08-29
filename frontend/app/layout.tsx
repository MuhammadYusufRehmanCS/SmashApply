import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "SmashApply",
  description: "Smash or Pass your job pipeline.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased">{children}</body>
    </html>
  );
}
