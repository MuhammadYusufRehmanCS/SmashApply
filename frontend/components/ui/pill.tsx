import type { ReactNode } from "react";

import { cn } from "@/lib/utils";

export type PillTone = "emerald" | "amber" | "red" | "zinc" | "indigo";

const TONE_CLASSES: Record<PillTone, string> = {
  emerald: "bg-emerald-500/10 text-emerald-400 border-emerald-500/20",
  amber: "bg-amber-500/10 text-amber-400 border-amber-500/20",
  red: "bg-red-500/10 text-red-400 border-red-500/20",
  zinc: "bg-zinc-800/50 text-zinc-500 border-zinc-700/50",
  indigo: "bg-indigo-500/10 text-indigo-400 border-indigo-500/20",
};

export function matchScoreTone(score: number): PillTone {
  if (score >= 75) return "emerald";
  if (score >= 50) return "amber";
  return "red";
}

export function Pill({
  tone,
  className,
  children,
}: {
  tone: PillTone;
  className?: string;
  children: ReactNode;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 whitespace-nowrap rounded-full border px-2 py-0.5 text-xs font-medium",
        TONE_CLASSES[tone],
        className
      )}
    >
      {children}
    </span>
  );
}
