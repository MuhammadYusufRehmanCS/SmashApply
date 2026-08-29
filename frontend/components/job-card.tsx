import { ExternalLink, Loader2 } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { cn } from "@/lib/utils";
import type { Job } from "@/types/job";

function matchBadgeVariant(score: number): "success" | "warning" | "muted" {
  if (score >= 80) return "success";
  if (score >= 50) return "warning";
  return "muted";
}

export function JobCard({
  job,
  onPass,
  onSmash,
  isSmashing,
}: {
  job: Job;
  onPass: (id: number) => void;
  onSmash: (id: number) => void;
  isSmashing: boolean;
}) {
  const isExpired = job.status === "expired";
  const applyUrl = job.final_url || job.raw_url;

  return (
    <Card className={cn("flex flex-col", isExpired && "opacity-60")}>
      <CardHeader className="flex-row items-start justify-between space-y-0">
        <div>
          <CardTitle>{job.title}</CardTitle>
          <p className="text-sm text-muted-foreground">{job.company}</p>
        </div>
        <Badge variant={matchBadgeVariant(job.match_score)}>{job.match_score}% match</Badge>
      </CardHeader>

      <CardContent className="flex-1 space-y-2">
        {isExpired && <Badge variant="warning">Expired — {job.validation_note ?? "link dead"}</Badge>}
        <p className="line-clamp-3 text-sm text-muted-foreground">
          {job.description || "No description captured yet."}
        </p>
        <a
          href={applyUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-1 text-sm text-primary hover:underline"
        >
          View posting <ExternalLink className="h-3.5 w-3.5" />
        </a>
      </CardContent>

      <CardFooter className="gap-2">
        <Button variant="pass" className="flex-1" onClick={() => onPass(job.id)}>
          Pass
        </Button>
        <Button
          variant="smash"
          className="flex-1"
          disabled={isExpired || isSmashing}
          onClick={() => onSmash(job.id)}
        >
          {isSmashing ? <Loader2 className="h-4 w-4 animate-spin" /> : "Smash"}
        </Button>
      </CardFooter>
    </Card>
  );
}
