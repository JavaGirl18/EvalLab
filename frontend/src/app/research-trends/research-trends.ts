import { Component, Input, OnInit, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ResearchScoredResult, ResearchHistoryEntry, BatchSummary, EvalService } from '../services/eval';

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

export interface DimensionStat {
  id: string;
  name: string;
  group: 'ii' | 'cf';
  avgSeverity: number;
  pctFlagged: number;    // severity >= 1
  pctModerate: number;   // severity >= 2
  totalEvals: number;
}

export interface ScorePoint {
  label: string;
  ii: number;
  cf: number;
}

export interface ModelPoint {
  model: string;
  ii: number;
  cf: number;
}

export interface DimensionRow {
  id: string;
  name: string;
  group: 'ii' | 'cf';
  cells: { model: string; severity: number; confidence: string; explanation: string }[];
}

export interface ModelComparisonMatrix {
  models: string[];
  iiRows: DimensionRow[];
  cfRows: DimensionRow[];
}

@Component({
  selector: 'app-research-trends',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './research-trends.html',
})
export class ResearchTrendsComponent implements OnInit {
  @Input() history: ResearchHistoryEntry[] = [];
  @Input() currentResults: ResearchScoredResult[] = [];

  activeSection: 'dimensions' | 'timeline' | 'models' = 'dimensions';
  modelView: 'matrix' | 'chart' = 'matrix';
  experimentFilter: string = '';

  // Batch filter state
  batches: BatchSummary[] = [];
  batchFilter: string = '';
  batchResults: ResearchScoredResult[] = [];
  batchLoading = false;
  batchError = '';

  constructor(private evalService: EvalService, private cdr: ChangeDetectorRef) {}

  ngOnInit() {
    this.evalService.listBatches().subscribe({
      next: (b) => {
        this.batches = b;
        const latest = b.find((x) => x.status === 'completed');
        if (latest && !this.batchFilter) {
          this.selectBatch(latest.batch_id);
        }
        this.cdr.detectChanges();
      },
      error: () => {},
    });
  }

  get filterMode(): 'batch' | 'experiment' {
    return this.batchFilter ? 'batch' : 'experiment';
  }

  selectBatch(batchId: string) {
    this.batchFilter      = batchId;
    this.experimentFilter = '';
    if (!batchId) {
      this.batchResults = [];
      return;
    }
    this.batchLoading = true;
    this.batchError   = '';
    this.evalService.getBatchResults(batchId).subscribe({
      next: (res) => {
        this.batchResults = res.results;
        this.batchLoading = false;
        this.cdr.detectChanges();
      },
      error: () => {
        this.batchError   = 'Could not load batch results.';
        this.batchLoading = false;
        this.cdr.detectChanges();
      },
    });
  }

  selectExperiment(expId: string) {
    this.experimentFilter = expId;
    this.batchFilter      = '';
    this.batchResults     = [];
  }

  get hasData(): boolean {
    return this.history.length > 0 || this.currentResults.length > 0 || this.batchResults.length > 0;
  }

  get experiments(): string[] {
    const ids = this.history
      .map((e) => e.experiment?.experiment_id)
      .filter((id): id is string => !!id);
    return [...new Set(ids)];
  }

  get filteredHistory(): ResearchHistoryEntry[] {
    if (this.filterMode === 'batch') return [];
    if (!this.experimentFilter) return this.history;
    return this.history.filter((e) => e.experiment?.experiment_id === this.experimentFilter);
  }

  get allResults(): ResearchScoredResult[] {
    if (this.filterMode === 'batch') return this.batchResults;
    return this.filteredHistory.flatMap((e) => e.results);
  }

  get dimensionStats(): DimensionStat[] {
    const source = this.allResults.length > 0 ? this.allResults : this.currentResults;
    if (source.length === 0) return [];

    return Object.entries(CATEGORY_META).map(([id, meta]) => {
      const scores = source
        .map((r) => r.dimension_scores[id]?.severity ?? 0);
      const total = scores.length;
      const flagged = scores.filter((s) => s >= 1).length;
      const moderate = scores.filter((s) => s >= 2).length;
      const avg = total > 0 ? scores.reduce((a, b) => a + b, 0) / total : 0;
      return {
        id,
        name: meta.name,
        group: meta.group,
        avgSeverity: Math.round(avg * 10) / 10,
        pctFlagged: total > 0 ? Math.round((flagged / total) * 100) : 0,
        pctModerate: total > 0 ? Math.round((moderate / total) * 100) : 0,
        totalEvals: total,
      };
    }).sort((a, b) => b.pctFlagged - a.pctFlagged);
  }

  get iiStats(): DimensionStat[] {
    return this.dimensionStats.filter((d) => d.group === 'ii');
  }

  get cfStats(): DimensionStat[] {
    return this.dimensionStats.filter((d) => d.group === 'cf');
  }

  get scoreTimeline(): ScorePoint[] {
    if (this.filterMode === 'batch') {
      // Group batch results by item_id to synthesize timeline points
      const grouped = new Map<string, ResearchScoredResult[]>();
      for (const r of this.batchResults) {
        const key = r.item_id || r.source_title;
        grouped.set(key, [...(grouped.get(key) ?? []), r]);
      }
      return [...grouped.entries()].map(([key, results]) => {
        const avgII = results.reduce((s, r) => s + r.overall_information_integrity_score, 0) / results.length;
        const avgCF = results.reduce((s, r) => s + r.overall_cultural_fidelity_score, 0) / results.length;
        const label = results[0].source_title;
        return {
          label: label.length > 28 ? label.slice(0, 28) + '…' : label,
          ii: Math.round(avgII * 10) / 10,
          cf: Math.round(avgCF * 10) / 10,
        };
      });
    }
    return [...this.filteredHistory]
      .reverse()
      .map((e) => ({
        label: e.source_title.length > 28 ? e.source_title.slice(0, 28) + '…' : e.source_title,
        ii: e.avg_ii_score,
        cf: e.avg_cf_score,
      }));
  }

  get modelComparison(): ModelPoint[] {
    const src = this.allResults.length > 0 ? this.allResults
      : this.currentResults.length > 0 ? this.currentResults
      : this.history[0]?.results ?? [];
    if (src.length === 0) return [];

    const byModel = new Map<string, { ii: number[]; cf: number[] }>();
    for (const r of src) {
      const m = r.model.split('/').pop() ?? r.model;
      if (!byModel.has(m)) byModel.set(m, { ii: [], cf: [] });
      byModel.get(m)!.ii.push(r.overall_information_integrity_score);
      byModel.get(m)!.cf.push(r.overall_cultural_fidelity_score);
    }
    return [...byModel.entries()].map(([model, scores]) => ({
      model,
      ii: Math.round(scores.ii.reduce((a, b) => a + b, 0) / scores.ii.length * 10) / 10,
      cf: Math.round(scores.cf.reduce((a, b) => a + b, 0) / scores.cf.length * 10) / 10,
    }));
  }

  get modelMatrix(): ModelComparisonMatrix {
    const src = this.allResults.length > 0 ? this.allResults
      : this.currentResults.length > 0 ? this.currentResults
      : this.history[0]?.results ?? [];
    if (src.length === 0) return { models: [], iiRows: [], cfRows: [] };

    // Preserve model order from first appearance
    const modelOrder: string[] = [];
    const seen = new Set<string>();
    for (const r of src) {
      const m = r.model.split('/').pop() ?? r.model;
      if (!seen.has(m)) { modelOrder.push(m); seen.add(m); }
    }

    const makeRows = (group: 'ii' | 'cf'): DimensionRow[] =>
      Object.entries(CATEGORY_META)
        .filter(([, meta]) => meta.group === group)
        .map(([id, meta]) => ({
          id,
          name: meta.name,
          group,
          cells: modelOrder.map((model) => {
            const mrs = src.filter((r) => (r.model.split('/').pop() ?? r.model) === model);
            const severities = mrs.map((r) => r.dimension_scores[id]?.severity ?? 0);
            const avg = severities.length
              ? Math.round(severities.reduce((a, b) => a + b, 0) / severities.length)
              : 0;
            const worst = mrs.reduce((best, r) =>
              (r.dimension_scores[id]?.severity ?? 0) > (best.dimension_scores[id]?.severity ?? 0) ? r : best,
              mrs[0]
            );
            return {
              model,
              severity: avg,
              confidence: worst?.dimension_scores[id]?.confidence ?? 'Low',
              explanation: worst?.dimension_scores[id]?.explanation ?? '',
            };
          }),
        }));

    return { models: modelOrder, iiRows: makeRows('ii'), cfRows: makeRows('cf') };
  }

  severityClass(s: number): string {
    return [
      'bg-green-100 text-green-700',
      'bg-yellow-100 text-yellow-700',
      'bg-orange-100 text-orange-700',
      'bg-red-100 text-red-700',
    ][s] ?? 'bg-gray-100 text-gray-500';
  }

  severityLabel(s: number): string {
    return ['0', '1', '2', '3'][s] ?? '—';
  }

  severityBarWidth(s: number): string {
    return s === 0 ? '4px' : `${(s / 3) * 100}%`;
  }

  modelColor(index: number): string {
    const colors = ['bg-indigo-400', 'bg-violet-400', 'bg-cyan-400', 'bg-pink-400', 'bg-amber-400'];
    return colors[index % colors.length];
  }

  modelTextColor(index: number): string {
    const colors = ['text-indigo-600', 'text-violet-600', 'text-cyan-600', 'text-pink-600', 'text-amber-600'];
    return colors[index % colors.length];
  }

  get evalCount(): number {
    return this.allResults.length || this.currentResults.length;
  }

  get activeFilterLabel(): string {
    if (this.filterMode === 'batch') {
      const b = this.batches.find((b) => b.batch_id === this.batchFilter);
      return b ? `Batch · ${b.experiment_id} (${b.completed}/${b.total} runs)` : this.batchFilter;
    }
    if (!this.experimentFilter) return 'All experiments';
    const meta = this.history.find((e) => e.experiment?.experiment_id === this.experimentFilter)?.experiment;
    if (!meta) return this.experimentFilter;
    return meta.experiment_name
      ? `${this.experimentFilter} · ${meta.experiment_name}`
      : this.experimentFilter;
  }

  get activeExperimentLabel(): string {
    return this.activeFilterLabel;
  }

  batchStatusColor(status: string): string {
    const map: Record<string, string> = {
      completed: 'text-green-600',
      running:   'text-blue-500',
      failed:    'text-red-500',
      pending:   'text-gray-400',
      cancelled: 'text-gray-400',
    };
    return map[status] ?? 'text-gray-400';
  }

  barWidth(pct: number): string {
    return `${Math.max(pct, 2)}%`;
  }

  barColor(pct: number): string {
    if (pct >= 75) return 'bg-red-400';
    if (pct >= 50) return 'bg-orange-400';
    if (pct >= 25) return 'bg-yellow-400';
    if (pct > 0)   return 'bg-yellow-300';
    return 'bg-green-300';
  }

  scoreBarWidth(score: number): string {
    return `${(score / 10) * 100}%`;
  }

  scoreBarColor(score: number): string {
    if (score >= 8) return 'bg-green-400';
    if (score >= 6) return 'bg-yellow-400';
    return 'bg-red-400';
  }

  scoreTextColor(score: number): string {
    if (score >= 8) return 'text-green-600';
    if (score >= 6) return 'text-yellow-500';
    return 'text-red-500';
  }
}
