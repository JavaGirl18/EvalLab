import { Component, Input, OnChanges, OnDestroy, OnInit, SimpleChanges, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { EvalService, BatchStatus, BatchSummary, SciqResults, SciqConditionSummary } from '../services/eval';

const POLL_MS = 4000;

@Component({
  selector: 'app-sciq-results',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './sciq-results.html',
})
export class SciqResultsComponent implements OnInit, OnChanges, OnDestroy {
  /** Batch to show; set when a SCIQ run is started from the sidebar. */
  @Input() batchId = '';

  batches: BatchSummary[] = [];
  selectedBatchId = '';
  status: BatchStatus | null = null;
  results: SciqResults | null = null;
  loading = false;
  error = '';
  itemsExpanded = false;

  private pollRef: ReturnType<typeof setInterval> | null = null;

  constructor(private evalService: EvalService, private cdr: ChangeDetectorRef) {}

  ngOnInit() {
    this.loadBatches();
  }

  ngOnChanges(changes: SimpleChanges) {
    if (changes['batchId'] && this.batchId && this.batchId !== this.selectedBatchId) {
      this.loadBatches();
      this.select(this.batchId);
    }
  }

  ngOnDestroy() {
    this.clearPoll();
  }

  loadBatches() {
    this.evalService.listBatches().subscribe({
      next: (all) => {
        this.batches = all.filter((b) => b.experiment_type === 'sciq');
        if (!this.selectedBatchId && this.batches.length > 0) {
          this.select(this.batchId || this.batches[0].batch_id);
        }
        this.cdr.detectChanges();
      },
      error: () => {
        this.error = 'Could not load batches.';
        this.cdr.detectChanges();
      },
    });
  }

  select(batchId: string) {
    this.clearPoll();
    this.selectedBatchId = batchId;
    this.status = null;
    this.results = null;
    this.error = '';
    this.itemsExpanded = false;
    if (!batchId) return;
    this.refreshStatus();
  }

  private refreshStatus() {
    const id = this.selectedBatchId;
    this.evalService.getBatchStatus(id).subscribe({
      next: (s) => {
        if (id !== this.selectedBatchId) return;
        this.status = s;
        if (s.status === 'running' || s.status === 'pending') {
          if (!this.pollRef) this.pollRef = setInterval(() => this.refreshStatus(), POLL_MS);
        } else {
          this.clearPoll();
          if (s.status === 'completed') this.loadResults();
          this.loadBatches();
        }
        this.cdr.detectChanges();
      },
      error: (err) => {
        this.clearPoll();
        this.error = err.error?.detail || 'Could not load batch status.';
        this.cdr.detectChanges();
      },
    });
  }

  private loadResults() {
    const id = this.selectedBatchId;
    this.loading = true;
    this.evalService.getSciqResults(id).subscribe({
      next: (r) => {
        if (id !== this.selectedBatchId) return;
        this.results = r;
        this.loading = false;
        this.cdr.detectChanges();
      },
      error: (err) => {
        this.loading = false;
        this.error = err.error?.detail || 'Could not load SCIQ results.';
        this.cdr.detectChanges();
      },
    });
  }

  private clearPoll() {
    if (this.pollRef !== null) {
      clearInterval(this.pollRef);
      this.pollRef = null;
    }
  }

  // ── Template helpers ────────────────────────────────────────────────────────

  get accurate(): SciqConditionSummary | null {
    return this.results?.summary.accurate_condition ?? null;
  }

  get erroneous(): SciqConditionSummary | null {
    return this.results?.summary.erroneous_condition ?? null;
  }

  get progressPct(): number {
    return this.status && this.status.total > 0 ? (this.status.completed / this.status.total) * 100 : 0;
  }

  pct(n: number, d: number): string {
    return d > 0 ? `${Math.round((n / d) * 100)}%` : '—';
  }

  formatP(p: number): string {
    if (p >= 0.001) return p.toFixed(3);
    return p.toExponential(1);
  }
}
