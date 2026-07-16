import { Component, OnInit, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { EvalService, ExternalPackage } from '../services/eval';

const ALLOWED_EXTENSIONS = new Set([
  '.json', '.md', '.txt', '.csv', '.html', '.htm',
  '.pdf', '.png', '.jpg', '.jpeg', '.gif', '.svg',
  '.mp3', '.wav', '.flac', '.ogg', '.m4a',
  '.sha256', '.sha512', '.md5',
]);

@Component({
  selector: 'app-external-packages',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './external-packages.html',
})
export class ExternalPackagesComponent implements OnInit {
  packages: ExternalPackage[] = [];
  selected: ExternalPackage | null = null;
  loading = false;
  error = '';

  // Upload state
  showUpload = false;
  sourceLabel = '';
  submittedBy = '';
  selectedFiles: File[] = [];
  rejectedFiles: string[] = [];
  uploading = false;
  uploadError = '';

  // Edit state
  editingMeta = false;
  draftMeta: Record<string, string> = {};
  draftNotes = '';
  saving = false;

  // Approve / reject state
  approving = false;
  rejecting = false;
  rejectReason = '';
  showRejectForm = false;
  actionResult = '';
  actionError = '';

  // Judge evaluation state
  showEvalForm = false;
  evalSourceFile = '';
  evalTransformationFile = '';
  evalSubjectLabel = '';
  evalTaskDescription = '';
  evaluating = false;
  evalResult: any = null;
  evalError = '';

  constructor(private evalService: EvalService, private cdr: ChangeDetectorRef) {}

  ngOnInit() { this.loadPackages(); }

  loadPackages() {
    this.loading = true;
    this.evalService.listExternalPackages().subscribe({
      next: (pkgs) => {
        this.packages = pkgs;
        if (this.selected) {
          this.selected = pkgs.find(p => p.pkg_id === this.selected!.pkg_id) ?? null;
        }
        this.loading = false;
        this.cdr.detectChanges();
      },
      error: () => { this.loading = false; this.cdr.detectChanges(); },
    });
  }

  selectPackage(pkg: ExternalPackage) {
    this.selected = pkg;
    this.editingMeta = false;
    this.showRejectForm = false;
    this.showEvalForm = false;
    this.actionResult = '';
    this.actionError = '';
    this.evalResult = null;
    this.evalError = '';
    this.draftMeta = { ...this.flatMeta(pkg) };
    this.draftNotes = pkg.notes || '';
    if (pkg.status === 'approved') {
      this.evalService.getExternalPackageResult(pkg.pkg_id).subscribe({
        next: (result) => { this.evalResult = result; this.cdr.detectChanges(); },
        error: () => {},
      });
    }
    this.cdr.detectChanges();
  }

  private flatMeta(pkg: ExternalPackage): Record<string, string> {
    const fields: Record<string, string> = {};
    const mapped = pkg.mapped_meta ?? {};
    for (const key of ['title', 'researcher', 'institution', 'methodology', 'description', 'date', 'contact']) {
      fields[key] = mapped[key] ?? pkg.detected_meta?.[key] ?? '';
    }
    return fields;
  }

  get detectedHighlights(): Array<{key: string; value: string}> {
    if (!this.selected) return [];
    const d = this.selected.detected_meta ?? {};
    return Object.entries(d)
      .filter(([k]) => !['manifest', 'readme_excerpt', 'file_count', 'file_types', 'has_manifest', 'has_readme', 'json_fields'].includes(k))
      .slice(0, 10)
      .map(([key, value]) => ({ key, value: typeof value === 'object' ? JSON.stringify(value).slice(0, 80) : String(value).slice(0, 120) }));
  }

  // ── File selection ──────────────────────────────────────────────────────────

  onFilesSelected(event: Event) {
    const input = event.target as HTMLInputElement;
    if (!input.files) return;
    this.rejectedFiles = [];
    this.selectedFiles = [];
    for (const file of Array.from(input.files)) {
      const ext = '.' + (file.name.split('.').pop() ?? '').toLowerCase();
      if (ALLOWED_EXTENSIONS.has(ext)) {
        this.selectedFiles.push(file);
      } else {
        this.rejectedFiles.push(this.displayName(file));
      }
    }
    this.cdr.detectChanges();
  }

  displayName(file: File): string {
    return (file as any).webkitRelativePath || file.name;
  }

  formatBytes(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  // ── Upload ──────────────────────────────────────────────────────────────────

  uploadPackage() {
    if (!this.selectedFiles.length) return;
    this.uploading = true;
    this.uploadError = '';
    const formData = new FormData();
    for (const file of this.selectedFiles) {
      formData.append('files', file, this.displayName(file));
    }
    formData.append('source_label', this.sourceLabel);
    formData.append('submitted_by', this.submittedBy);

    this.evalService.uploadExternalPackage(formData).subscribe({
      next: (pkg) => {
        this.uploading = false;
        this.showUpload = false;
        this.selectedFiles = [];
        this.sourceLabel = '';
        this.submittedBy = '';
        this.packages = [pkg, ...this.packages];
        this.selectPackage(pkg);
        this.cdr.detectChanges();
      },
      error: (err) => {
        this.uploadError = err.error?.detail || 'Upload failed.';
        this.uploading = false;
        this.cdr.detectChanges();
      },
    });
  }

  // ── Save mapped metadata ────────────────────────────────────────────────────

  saveMeta() {
    if (!this.selected) return;
    this.saving = true;
    const clean: Record<string, string> = {};
    for (const [k, v] of Object.entries(this.draftMeta)) {
      if (v.trim()) clean[k] = v.trim();
    }
    this.evalService.updateExternalPackageMeta(this.selected.pkg_id, clean, this.draftNotes).subscribe({
      next: (pkg) => {
        this.selected = pkg;
        this.packages = this.packages.map(p => p.pkg_id === pkg.pkg_id ? pkg : p);
        this.editingMeta = false;
        this.saving = false;
        this.cdr.detectChanges();
      },
      error: () => { this.saving = false; this.cdr.detectChanges(); },
    });
  }

  // ── Approve ─────────────────────────────────────────────────────────────────

  approve() {
    if (!this.selected) return;
    this.approving = true;
    this.actionError = '';
    this.evalService.approveExternalPackage(this.selected.pkg_id).subscribe({
      next: (res) => {
        this.approving = false;
        this.actionResult = `Evaluation Record created: ${res.evaluation_id}`;
        this.loadPackages();
        this.cdr.detectChanges();
      },
      error: (err) => {
        this.actionError = err.error?.detail || 'Approval failed.';
        this.approving = false;
        this.cdr.detectChanges();
      },
    });
  }

  // ── Reject ──────────────────────────────────────────────────────────────────

  reject() {
    if (!this.selected) return;
    this.rejecting = true;
    this.actionError = '';
    this.evalService.rejectExternalPackage(this.selected.pkg_id, this.rejectReason).subscribe({
      next: (pkg) => {
        this.selected = pkg;
        this.packages = this.packages.map(p => p.pkg_id === pkg.pkg_id ? pkg : p);
        this.rejecting = false;
        this.showRejectForm = false;
        this.rejectReason = '';
        this.cdr.detectChanges();
      },
      error: (err) => {
        this.actionError = err.error?.detail || 'Rejection failed.';
        this.rejecting = false;
        this.cdr.detectChanges();
      },
    });
  }

  // ── Helpers ─────────────────────────────────────────────────────────────────

  statusColor(status: string): string {
    return ({
      imported: 'bg-blue-100 text-blue-700',
      reviewed: 'bg-amber-100 text-amber-700',
      approved: 'bg-green-100 text-green-700',
      rejected: 'bg-red-100 text-red-600',
    } as Record<string, string>)[status] ?? 'bg-gray-100 text-gray-500';
  }

  formatDate(iso: string | null): string {
    if (!iso) return '—';
    return new Date(iso).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
  }

  openRerun() {
    if (this.evalResult) {
      this.evalSourceFile = this.evalResult.source_file || '';
      this.evalTransformationFile = this.evalResult.transformation_file || '';
      this.evalSubjectLabel = this.evalResult.subject_label || '';
      this.evalTaskDescription = this.evalResult.task_description || '';
    }
    this.showEvalForm = true;
  }

  runEvaluation() {
    if (!this.selected || !this.evalSourceFile || !this.evalTransformationFile) return;
    this.evaluating = true;
    this.evalResult = null;
    this.evalError = '';
    this.evalService.evaluateExternalPackage(this.selected.pkg_id, {
      source_file:         this.evalSourceFile,
      transformation_file: this.evalTransformationFile,
      subject_label:       this.evalSubjectLabel || 'External System',
      task_description:    this.evalTaskDescription || 'Reproduce the source text accurately.',
    }).subscribe({
      next: (result) => {
        this.evalResult = result;
        this.evaluating = false;
        this.showEvalForm = false;
        this.cdr.detectChanges();
      },
      error: (err) => {
        this.evalError = err.error?.detail || 'Evaluation failed.';
        this.evaluating = false;
        this.cdr.detectChanges();
      },
    });
  }

  scoreColor(score: number): string {
    if (score >= 7) return 'text-green-600';
    if (score >= 4) return 'text-amber-500';
    return 'text-red-500';
  }

  severityLabel(s: number): string {
    return ['Not Present', 'Minor', 'Moderate', 'Severe'][s] ?? `${s}`;
  }

  severityChipClass(s: number): string {
    if (s >= 3) return 'bg-red-100 text-red-700';
    if (s === 2) return 'bg-amber-100 text-amber-700';
    return 'bg-yellow-50 text-yellow-600';
  }

  downloadResult(format: 'html' | 'json' = 'html') {
    if (!this.selected) return;
    const pkg = this.selected;
    const ext = format === 'html' ? 'html' : 'json';
    const filename = `${pkg.pkg_id}_${pkg.evaluation_id ?? 'eval'}_evaluation_record.${ext}`;
    this.evalService.exportExternalPackageRecord(pkg.pkg_id, format).subscribe({
      next: (blob) => {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filename;
        a.click();
        URL.revokeObjectURL(url);
      },
      error: () => {},
    });
  }

  get evalFailureDetails(): Array<{id: string; severity: number; confidence: string; explanation: string; source_evidence: string; output_evidence: string}> {
    if (!this.evalResult?.dimension_scores) return [];
    return (Object.entries(this.evalResult.dimension_scores) as [string, any][])
      .filter(([, d]) => d.severity > 0)
      .sort(([, a], [, b]) => b.severity - a.severity)
      .map(([id, d]) => ({ id, ...d }));
  }

  lifecycleSteps = ['imported', 'reviewed', 'approved'];
  stepDone(pkg: ExternalPackage, step: string): boolean {
    const order = ['imported', 'reviewed', 'approved'];
    return order.indexOf(pkg.status) >= order.indexOf(step);
  }
}
