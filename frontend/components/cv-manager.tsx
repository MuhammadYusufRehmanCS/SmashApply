"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { UploadCloud } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { api } from "@/lib/api";

const ACCEPTED_EXTENSIONS = [".pdf", ".docx"];

export function CvManager({ onClose }: { onClose: () => void }) {
  const [text, setText] = useState("");
  const [saving, setSaving] = useState(false);
  const [loaded, setLoaded] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [uploadedFileName, setUploadedFileName] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    api
      .getCv()
      .then((cv) => setText(cv?.raw_text ?? ""))
      .finally(() => setLoaded(true));
  }, []);

  async function handleSave() {
    setSaving(true);
    try {
      await api.uploadCv(text);
      onClose();
    } finally {
      setSaving(false);
    }
  }

  const handleFile = useCallback(async (file: File) => {
    const isAccepted = ACCEPTED_EXTENSIONS.some((ext) => file.name.toLowerCase().endsWith(ext));
    if (!isAccepted) {
      setUploadError("Only .pdf and .docx files are supported.");
      return;
    }
    setUploadError(null);
    setUploading(true);
    try {
      const cv = await api.uploadCvFile(file);
      setText(cv.raw_text);
      setUploadedFileName(file.name);
    } catch (e) {
      setUploadError(String(e));
    } finally {
      setUploading(false);
    }
  }, []);

  function handleDrop(e: React.DragEvent<HTMLDivElement>) {
    e.preventDefault();
    setDragActive(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4">
      <Card className="flex max-h-[85vh] w-full max-w-2xl flex-col">
        <CardHeader>
          <CardTitle>Master CV</CardTitle>
          <p className="text-sm text-muted-foreground">
            Drag and drop a .pdf or .docx file, or paste your CV text directly. This is the
            source of truth for vector matching and tailoring — it is never rewritten, only
            referenced.
          </p>
        </CardHeader>
        <CardContent className="flex-1 space-y-3 overflow-y-auto">
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDragActive(true);
            }}
            onDragLeave={() => setDragActive(false)}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`flex cursor-pointer flex-col items-center justify-center gap-2 rounded-md border-2 border-dashed p-6 text-center text-sm transition-colors ${
              dragActive ? "border-primary bg-primary/5" : "border-border"
            }`}
          >
            <UploadCloud className="h-6 w-6 text-muted-foreground" />
            <p className="text-muted-foreground">
              {uploading
                ? "Extracting text..."
                : uploadedFileName
                  ? `Loaded "${uploadedFileName}" — drop another file to replace`
                  : "Drag & drop a .pdf or .docx file here, or click to browse"}
            </p>
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf,.docx"
              className="hidden"
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) handleFile(file);
                e.target.value = "";
              }}
            />
          </div>
          {uploadError && <p className="text-sm text-pass">{uploadError}</p>}
          <textarea
            className="h-64 w-full resize-none rounded-md border border-border bg-background p-3 text-sm outline-none focus:ring-2 focus:ring-primary"
            placeholder={loaded ? "Paste your CV text here..." : "Loading..."}
            value={text}
            onChange={(e) => setText(e.target.value)}
          />
        </CardContent>
        <CardFooter className="justify-end gap-2">
          <Button variant="outline" onClick={onClose}>
            Cancel
          </Button>
          <Button onClick={handleSave} disabled={saving || !text.trim()}>
            {saving ? "Saving..." : "Save CV"}
          </Button>
        </CardFooter>
      </Card>
    </div>
  );
}
