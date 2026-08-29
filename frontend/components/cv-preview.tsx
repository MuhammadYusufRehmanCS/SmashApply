"use client";

import { useState } from "react";
import { Download, X } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";

export function CvPreview({
  tailoredCv,
  jobTitle,
  onClose,
}: {
  tailoredCv: string;
  jobTitle: string;
  onClose: () => void;
}) {
  const [exporting, setExporting] = useState(false);

  async function handleExportPdf() {
    setExporting(true);
    try {
      const { pdf } = await import("@react-pdf/renderer");
      const { CvPdfDocument } = await import("@/components/cv-pdf-document");
      const blob = await pdf(<CvPdfDocument text={tailoredCv} />).toBlob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `CV - ${jobTitle}.pdf`;
      a.click();
      URL.revokeObjectURL(url);
    } finally {
      setExporting(false);
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
      <Card className="flex max-h-[85vh] w-full max-w-2xl flex-col">
        <CardHeader className="flex-row items-center justify-between space-y-0">
          <CardTitle>Tailored CV — {jobTitle}</CardTitle>
          <Button variant="ghost" size="icon" onClick={onClose}>
            <X className="h-4 w-4" />
          </Button>
        </CardHeader>
        <CardContent className="flex-1 overflow-y-auto">
          <pre className="whitespace-pre-wrap font-sans text-sm text-foreground">{tailoredCv}</pre>
        </CardContent>
        <CardFooter className="justify-end gap-2">
          <Button variant="outline" onClick={onClose}>
            Close
          </Button>
          <Button onClick={handleExportPdf} disabled={exporting}>
            <Download className="h-4 w-4" />
            {exporting ? "Exporting..." : "Export PDF"}
          </Button>
        </CardFooter>
      </Card>
    </div>
  );
}
