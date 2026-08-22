import { Component, Input, OnInit, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { forkJoin } from 'rxjs';
import { ResearchScoredResult, DimensionScore, BatchResults, BatchSummary, EvalService } from '../services/eval';

// Taxonomy display metadata — update if taxonomy.yaml categories change.
const CATEGORY_META: Record<string, { name: string; group: 'ii' | 'cf' }> = {
  FH:  { name: 'Factual Hallucination',      group: 'ii' },
  CXH: { name: 'Contextual Hallucination',   group: 'ii' },
  CTH: { name: 'Cultural Hallucination',     group: 'ii' },
  AH:  { name: 'Attribution Hallucination',  group: 'ii' },
  FS:  { name: 'False Statement',            group: 'ii' },
  UC:  { name: 'Unsupported Claim',          group: 'ii' },
  CL:  { name: 'Context Loss',               group: 'ii' },
  EX:  { name: 'Exaggeration',               group: 'ii' },
  OC:  { name: 'Overconfidence',             group: 'ii' },
  AL:  { name: 'Attribution Loss',           group: 'ii' },
  CM:  { name: 'Cultural Misrepresentation', group: 'cf' },
  FB:  { name: 'Framing Bias',               group: 'cf' },
  PB:  { name: 'Political Bias',             group: 'cf' },
  ST:  { name: 'Stereotyping',               group: 'cf' },
};

export interface DimensionRow {
  id: string;
  name: string;
  score: DimensionScore;
}

@Component({
  selector: 'app-research-results',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './research-results.html',
})
export class ResearchResultsComponent implements OnInit {
  @Input() results: ResearchScoredResult[] = [];
  @Input() loading = false;

  showTaxonomyKey = false;

  // Batch/experiment filter state
  batches: BatchSummary[] = [];
  batchFilter = '';
  experimentFilter = '';
  batchData: BatchResults | null = null;
  batchLoading = false;
  expandedArticles: Record<string, boolean> = {};

  constructor(private evalService: EvalService, private cdr: ChangeDetectorRef) {}

  ngOnInit() {
    this.evalService.listBatches().subscribe({
      next: (batches) => {
        this.batches = batches;
        if (this.results.length > 0) { this.cdr.detectChanges(); return; }
        const latest = batches.find((b) => b.status === 'completed' && b.completed > 0);
        if (!latest) { this.batchLoading = false; this.cdr.detectChanges(); return; }
        this.selectBatch(latest.batch_id);
      },
      error: () => { this.batchLoading = false; this.cdr.detectChanges(); },
    });
  }

  get experiments(): string[] {
    const ids = this.batches
      .filter((b) => b.status === 'completed' && b.completed > 0)
      .map((b) => b.experiment_id);
    return [...new Set(ids)];
  }

  get filteredBatches(): BatchSummary[] {
    const completed = this.batches.filter((b) => b.completed > 0);
    return this.experimentFilter
      ? completed.filter((b) => b.experiment_id === this.experimentFilter)
      : completed;
  }

  selectBatch(batchId: string) {
    this.batchFilter = batchId;
    if (!batchId) {
      if (this.experimentFilter) {
        this.selectExperiment(this.experimentFilter);
      } else {
        this.batchData = null;
      }
      return;
    }
    const batch = this.batches.find((b) => b.batch_id === batchId);
    if (batch && batch.experiment_id !== this.experimentFilter) {
      this.experimentFilter = batch.experiment_id;
    }
    this.batchLoading = true;
    this.evalService.getBatchResults(batchId).subscribe({
      next: (data) => {
        this.batchData = data;
        const firstId = data.results[0]?.item_id;
        if (firstId) this.expandedArticles = { [firstId]: true };
        this.batchLoading = false;
        this.cdr.detectChanges();
      },
      error: () => { this.batchLoading = false; this.cdr.detectChanges(); },
    });
  }

  selectExperiment(expId: string) {
    this.experimentFilter = expId;
    this.batchFilter      = '';
    this.batchData        = null;
    if (!expId) return;

    const experimentBatches = this.batches.filter(
      (b) => b.experiment_id === expId && b.status === 'completed' && b.completed > 0
    );
    if (experimentBatches.length === 0) return;

    this.batchLoading = true;
    forkJoin(experimentBatches.map((b) => this.evalService.getBatchResults(b.batch_id))).subscribe({
      next: (responses) => {
        const allResults = responses.flatMap((r) => r.results);
        this.batchData = {
          batch_id: expId,
          experiment_id: expId,
          experiment_meta: responses[0].experiment_meta,
          results: allResults,
          summary: this.computeSummary(allResults, responses.reduce((s, r) => s + r.summary.expected, 0)),
        };
        const firstId = allResults[0]?.item_id;
        if (firstId) this.expandedArticles = { [firstId]: true };
        this.batchLoading = false;
        this.cdr.detectChanges();
      },
      error: () => { this.batchLoading = false; this.cdr.detectChanges(); },
    });
  }

  private computeSummary(results: ResearchScoredResult[], expected: number): BatchResults['summary'] {
    const byModel = new Map<string, { ii: number[]; cf: number[] }>();
    for (const r of results) {
      if (!byModel.has(r.model)) byModel.set(r.model, { ii: [], cf: [] });
      byModel.get(r.model)!.ii.push(r.overall_information_integrity_score);
      byModel.get(r.model)!.cf.push(r.overall_cultural_fidelity_score);
    }
    const avg_ii_by_model: Record<string, number> = {};
    const avg_cf_by_model: Record<string, number> = {};
    for (const [m, scores] of byModel) {
      avg_ii_by_model[m] = Math.round(scores.ii.reduce((a, b) => a + b, 0) / scores.ii.length * 10) / 10;
      avg_cf_by_model[m] = Math.round(scores.cf.reduce((a, b) => a + b, 0) / scores.cf.length * 10) / 10;
    }
    const catCounts = new Map<string, number>();
    for (const r of results) {
      for (const cat of r.most_significant_failures) catCounts.set(cat, (catCounts.get(cat) ?? 0) + 1);
    }
    const top_failure_categories: [string, number][] = [...catCounts.entries()]
      .sort((a, b) => b[1] - a[1]).slice(0, 5);
    return { expected, completed: results.length, failed: 0, skipped: 0, pending: 0,
             avg_ii_by_model, avg_cf_by_model, top_failure_categories };
  }

  get batchModelNames(): string[] {
    if (!this.batchData) return [];
    return Object.keys(this.batchData.summary.avg_ii_by_model);
  }

  get batchArticles(): { itemId: string; title: string; results: ResearchScoredResult[] }[] {
    if (!this.batchData?.results?.length) return [];
    const grouped = new Map<string, ResearchScoredResult[]>();
    for (const r of this.batchData.results) {
      const key = r.item_id || 'unknown';
      grouped.set(key, [...(grouped.get(key) ?? []), r]);
    }
    return [...grouped.entries()].map(([itemId, results]) => ({
      itemId,
      title: results[0]?.source_title ?? itemId,
      results,
    }));
  }

  toggleArticle(id: string) {
    this.expandedArticles = { ...this.expandedArticles, [id]: !this.expandedArticles[id] };
  }

  isArticleExpanded(id: string): boolean {
    return !!this.expandedArticles[id];
  }

  modelDetailExpanded: Record<string, boolean> = {};

  toggleModelDetail(articleId: string, model: string): void {
    const key = `${articleId}::${model}`;
    this.modelDetailExpanded = { ...this.modelDetailExpanded, [key]: !this.modelDetailExpanded[key] };
  }

  isModelDetailExpanded(articleId: string, model: string): boolean {
    return !!this.modelDetailExpanded[`${articleId}::${model}`];
  }

  articleFailureSummary(article: { itemId: string; title: string; results: ResearchScoredResult[] }): string[] {
    const seen = new Set<string>();
    for (const r of article.results) {
      for (const cat of r.most_significant_failures) seen.add(cat);
    }
    return [...seen];
  }

  readonly TAXONOMY_KEY = Object.entries(CATEGORY_META).map(([id, meta]) => ({
    id,
    name: meta.name,
    group: meta.group === 'ii' ? 'Information Integrity' : 'Cultural Fidelity',
  }));

  readonly TRUNCATE = 400;
  readonly skeletonCards = Array(3);

  responseExpanded: Record<string, boolean> = {};
  evidenceExpanded: Record<string, boolean> = {};

  get contextResult(): ResearchScoredResult | null {
    return this.results[0] ?? null;
  }

  get variantLabel(): string {
    return (this.contextResult?.prompt_variant_name ?? '').replace(/_/g, ' ');
  }

  iiRows(result: ResearchScoredResult): DimensionRow[] {
    return this.rowsForGroup(result, 'ii');
  }

  cfRows(result: ResearchScoredResult): DimensionRow[] {
    return this.rowsForGroup(result, 'cf');
  }

  private rowsForGroup(result: ResearchScoredResult, group: 'ii' | 'cf'): DimensionRow[] {
    return Object.entries(CATEGORY_META)
      .filter(([, meta]) => meta.group === group)
      .map(([id, meta]) => ({
        id,
        name: meta.name,
        score: result.dimension_scores[id] ?? {
          severity: 0, confidence: 'Low',
          explanation: 'Not evaluated.',
          source_evidence: 'N/A', output_evidence: 'N/A',
        },
      }));
  }

  hasSeverity(result: ResearchScoredResult): boolean {
    return Object.values(result.dimension_scores).some((d) => d.severity > 0);
  }

  // ── Display helpers ──────────────────────────────────────────────────────────

  severityLabel(s: number): string {
    return ['Not Present', 'Minor', 'Moderate', 'Severe'][s] ?? '—';
  }

  severityClass(s: number): string {
    return [
      'bg-green-100 text-green-700',
      'bg-yellow-100 text-yellow-700',
      'bg-orange-100 text-orange-700',
      'bg-red-100 text-red-700',
    ][s] ?? 'bg-gray-100 text-gray-600';
  }

  scoreColor(score: number): string {
    if (score >= 8) return 'text-green-600';
    if (score >= 6) return 'text-yellow-500';
    return 'text-red-500';
  }

  confidenceClass(c: string): string {
    if (c === 'High')   return 'bg-blue-100 text-blue-700';
    if (c === 'Medium') return 'bg-gray-100 text-gray-600';
    return 'bg-gray-50 text-gray-400';
  }

  // ── Expand/collapse ──────────────────────────────────────────────────────────

  toggleResponse(model: string): void {
    this.responseExpanded = { ...this.responseExpanded, [model]: !this.responseExpanded[model] };
  }

  isResponseExpanded(model: string): boolean {
    return !!this.responseExpanded[model];
  }

  displayText(result: ResearchScoredResult): string {
    if (this.isResponseExpanded(result.model) || result.response_text.length <= this.TRUNCATE) {
      return result.response_text;
    }
    return result.response_text.slice(0, this.TRUNCATE) + '…';
  }

  toggleEvidence(model: string, catId: string): void {
    const key = `${model}-${catId}`;
    this.evidenceExpanded = { ...this.evidenceExpanded, [key]: !this.evidenceExpanded[key] };
  }

  isEvidenceExpanded(model: string, catId: string): boolean {
    return !!this.evidenceExpanded[`${model}-${catId}`];
  }

  formatTimestamp(ts: string): string {
    if (!ts) return '';
    return new Date(ts).toLocaleString(undefined, {
      month: 'short', day: 'numeric',
      hour: '2-digit', minute: '2-digit',
    });
  }

  exportRecord(result: ResearchScoredResult): void {
    if (!this.batchData) return;
    const runKey = [
      this.batchData.experiment_id,
      result.item_id,
      result.prompt_variant_name,
      result.model,
    ].join('|');
    this.evalService.exportEvalRecordByRunKey(runKey);
  }
}
