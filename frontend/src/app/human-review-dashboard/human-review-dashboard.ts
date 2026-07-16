import {
  Component, Input, OnInit, OnDestroy, OnChanges, SimpleChanges,
  ChangeDetectorRef,
} from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import {
  EvalService,
  HumanReviewSummary,
  HumanReviewStats,
  HumanReviewDetail,
  HumanReviewResponse,
  EmergingPatterns,
  HumanReviewCounts,
} from '../services/eval';

interface LifecycleStep {
  label: string;
  done: boolean;
  ts: string | null;
}

const PLAIN_LABELS: Record<string, string> = {
  FH:  'Fabricated / Hallucinated Facts',
  CXH: 'Context Hallucination',
  CTH: 'Cultural Translation Hallucination',
  AH:  'Attribution Hallucination',
  FS:  'False Specificity',
  UC:  'Unsupported Claims',
  CL:  'Cultural Loss',
  EX:  'Exoticization',
  OC:  'Othering / Community Framing',
  AL:  'Agency / Leadership Erasure',
  CM:  'Community Voice Marginalization',
  FB:  'Framing Bias',
  PB:  'Political / Ideological Bias',
  ST:  'Stereotyping',
};

const DISAGREE_LABELS: Record<string, string> = {
  failure_not_present:          'Failure not present',
  failure_category_incorrect:   'Wrong category',
  severity_too_high:            'Severity too high',
  severity_too_low:             'Severity too low',
  important_failure_missing:    'Important failure missing',
  judge_explanation_inaccurate: 'Inaccurate explanation',
  source_context_misunderstood: 'Source context misunderstood',
  other:                        'Other',
};

interface ReviewGroup {
  item_id: string;
  title: string;
  reviews: HumanReviewSummary[];
}

@Component({
  selector: 'app-human-review-dashboard',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './human-review-dashboard.html',
})
export class HumanReviewDashboardComponent implements OnInit, OnDestroy, OnChanges {
  @Input() experimentId = '';

  // Scope
  filterMode: 'experiment' | 'batch' = 'experiment';
  batchId = '';
  reviewRound: number | null = null;

  // Data
  counts: HumanReviewCounts | null = null;
  reviews: HumanReviewSummary[] = [];
  stats: HumanReviewStats | null = null;
  patterns: EmergingPatterns | null = null;

  // Loading states
  loadingReviews  = false;
  loadingStats    = false;
  loadingPatterns = false;
  error: string | null = null;

  // Detail panel
  detailReview: HumanReviewDetail | null = null;
  detailResponses: HumanReviewResponse[] = [];
  detailLoading = false;
  showDetail = false;

  // Active stat sub-tab
  statsTab: 'category' | 'severity' | 'reviewer' = 'category';

  private refreshTimer: ReturnType<typeof setInterval> | null = null;

  constructor(private svc: EvalService, private cdr: ChangeDetectorRef) {}

  ngOnInit() {
    this.load();
    this.refreshTimer = setInterval(() => this.load(), 30_000);
  }

  ngOnDestroy() {
    if (this.refreshTimer) clearInterval(this.refreshTimer);
  }

  ngOnChanges(changes: SimpleChanges) {
    if (changes['experimentId'] && !changes['experimentId'].firstChange) {
      this.load();
    }
  }

  private get queryParams() {
    const p: { batch_id?: string; experiment_id?: string; review_round?: number } = {};
    if (this.filterMode === 'batch' && this.batchId) p.batch_id = this.batchId;
    else if (this.experimentId)                       p.experiment_id = this.experimentId;
    if (this.reviewRound) p.review_round = this.reviewRound;
    return p;
  }

  load() {
    this.loadReviews();
    this.loadStats();
    this.loadPatterns();
  }

  loadReviews() {
    const p = this.queryParams;
    if (!p.batch_id && !p.experiment_id) return;
    this.loadingReviews = true;
    this.error = null;
    this.svc.listHumanReviews(p).subscribe({
      next: (data) => { this.reviews = data; this.loadingReviews = false; this.cdr.markForCheck(); },
      error: (e) => { this.error = e.message; this.loadingReviews = false; this.cdr.markForCheck(); },
    });
    if (p.batch_id) {
      this.svc.countHumanReviews(p.batch_id).subscribe({
        next: (c) => { this.counts = c; this.cdr.markForCheck(); },
      });
    }
  }

  loadStats() {
    const p = this.queryParams;
    if (!p.batch_id && !p.experiment_id) return;
    this.loadingStats = true;
    this.svc.getHumanReviewSummary(p).subscribe({
      next: (s) => { this.stats = s; this.loadingStats = false; this.cdr.markForCheck(); },
      error: () => { this.loadingStats = false; this.cdr.markForCheck(); },
    });
  }

  loadPatterns() {
    const p = this.queryParams;
    if (!p.batch_id && !p.experiment_id) return;
    this.loadingPatterns = true;
    this.svc.getHumanReviewPatterns(p).subscribe({
      next: (pt) => { this.patterns = pt; this.loadingPatterns = false; this.cdr.markForCheck(); },
      error: () => { this.loadingPatterns = false; this.cdr.markForCheck(); },
    });
  }

  get reviewGroups(): ReviewGroup[] {
    const map = new Map<string, ReviewGroup>();
    for (const r of this.reviews) {
      if (!map.has(r.dataset_item_id)) {
        map.set(r.dataset_item_id, {
          item_id: r.dataset_item_id,
          title:   r.source_title || r.dataset_item_id,
          reviews: [],
        });
      }
      map.get(r.dataset_item_id)!.reviews.push(r);
    }
    return Array.from(map.values());
  }

  get completionPct(): number {
    if (!this.stats || !this.stats.total_assigned) return 0;
    return Math.round(this.stats.completion_rate * 100);
  }

  get agreementPct(): number {
    if (!this.stats) return 0;
    return Math.round(this.stats.agreement.yes_pct * 100);
  }

  get partialPct(): number {
    if (!this.stats) return 0;
    return Math.round(this.stats.agreement.partially_pct * 100);
  }

  get disagreePct(): number {
    if (!this.stats) return 0;
    return Math.round(this.stats.agreement.no_pct * 100);
  }

  statusColor(status: string): string {
    return {
      pending:   'bg-gray-100 text-gray-600',
      exported:  'bg-blue-100 text-blue-700',
      completed: 'bg-green-100 text-green-700',
      archived:  'bg-yellow-100 text-yellow-700',
    }[status] ?? 'bg-gray-100 text-gray-500';
  }

  categoryLabel(id: string): string {
    return PLAIN_LABELS[id] || id;
  }

  disagreeLabel(key: string): string {
    return DISAGREE_LABELS[key] || key;
  }

  agreePctForBucket(bucket: Record<string, number>): number {
    const total = bucket['total'] || 0;
    if (!total) return 0;
    return Math.round(((bucket['yes'] || 0) / total) * 100);
  }

  barWidth(pct: number): string {
    return `${Math.max(2, Math.min(100, pct))}%`;
  }

  barColor(pct: number): string {
    if (pct >= 70) return 'bg-green-400';
    if (pct >= 40) return 'bg-yellow-400';
    return 'bg-red-400';
  }

  openDetail(reviewId: string) {
    this.showDetail = true;
    this.detailLoading = true;
    this.detailReview = null;
    this.detailResponses = [];
    this.svc.getHumanReviewDetail(reviewId).subscribe({
      next: (d) => { this.detailReview = d; this.detailLoading = false; this.cdr.markForCheck(); },
      error: () => { this.detailLoading = false; this.cdr.markForCheck(); },
    });
    this.svc.getHumanReviewResponses(reviewId).subscribe({
      next: (rs) => { this.detailResponses = rs; this.cdr.markForCheck(); },
    });
  }

  closeDetail() {
    this.showDetail = false;
    this.detailReview = null;
    this.detailResponses = [];
  }

  downloadHtml(reviewId: string) {
    window.open(this.svc.exportHumanReviewUrl('html', { review_id: reviewId }), '_blank');
  }

  downloadJson() {
    window.open(this.svc.exportHumanReviewUrl('json', this.queryParams), '_blank');
  }

  downloadContextCsv() {
    window.open(this.svc.exportHumanReviewUrl('csv_context', this.queryParams), '_blank');
  }

  onFilterChange() {
    this.batchId = '';
    this.load();
  }

  onBatchIdChange() {
    this.load();
  }

  categoryKeys(obj: Record<string, Record<string, number>>): string[] {
    return Object.keys(obj).sort((a, b) =>
      this.agreePctForBucket(obj[b]) - this.agreePctForBucket(obj[a])
    );
  }

  severityOrder = ['3', '2', '1', '0'];
  severityLabel(k: string): string {
    return { '3': 'Severe (3)', '2': 'Moderate (2)', '1': 'Minor (1)', '0': 'Clean (0)' }[k] || k;
  }

  topDisagreePct(n: number, total: number): number {
    return total ? Math.round((n / total) * 100) : 0;
  }

  lifecycleSteps(review: HumanReviewDetail, responses: HumanReviewResponse[]): LifecycleStep[] {
    const importedAt = responses.length > 0
      ? responses.reduce((earliest, r) =>
          !earliest || r.imported_at < earliest ? r.imported_at : earliest,
          null as string | null)
      : null;
    return [
      { label: 'Generated',              done: true,                              ts: review.created_at },
      { label: 'Exported',               done: !!review.exported_at,              ts: review.exported_at ?? null },
      { label: 'Completed by reviewer',  done: !!review.completed_at,             ts: review.completed_at ?? null },
      { label: 'Responses imported',     done: !!importedAt,                      ts: importedAt },
      { label: 'Archived',               done: review.review_status === 'archived', ts: null },
    ];
  }

  exportEvalRecord(runKey: string) {
    if (!runKey) return;
    this.svc.exportEvalRecordByRunKey(runKey);
  }
}
