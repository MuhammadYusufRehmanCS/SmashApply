import { Flame, ThumbsDown, Zap } from "lucide-react";

import { Card, CardContent } from "@/components/ui/card";
import type { Metrics } from "@/types/job";

function MetricTile({
  icon,
  label,
  value,
}: {
  icon: React.ReactNode;
  label: string;
  value: number;
}) {
  return (
    <Card>
      <CardContent className="flex items-center gap-4 p-5">
        <div className="flex h-10 w-10 items-center justify-center rounded-full bg-muted">
          {icon}
        </div>
        <div>
          <p className="text-2xl font-bold leading-none">{value}</p>
          <p className="text-sm text-muted-foreground">{label}</p>
        </div>
      </CardContent>
    </Card>
  );
}

export function MetricsBar({ metrics }: { metrics: Metrics | null }) {
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-3">
      <MetricTile
        icon={<Flame className="h-5 w-5 text-smash" />}
        label="Smashed (Applied)"
        value={metrics?.smashed ?? 0}
      />
      <MetricTile
        icon={<ThumbsDown className="h-5 w-5 text-pass" />}
        label="Passed"
        value={metrics?.passed ?? 0}
      />
      <MetricTile
        icon={<Zap className="h-5 w-5 text-primary" />}
        label="Active Pipeline"
        value={metrics?.active_pipeline ?? 0}
      />
    </div>
  );
}
