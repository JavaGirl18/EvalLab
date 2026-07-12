import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { ResearchScoredResult, ResearchHistoryEntry } from '../services/eval';

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
export class ResearchTrendsComponent {
  @Input() history: ResearchHistoryEntry[] = [];
  @Input() currentResults: ResearchScoredResult[] = [];

  activeSection: 'dimensions' | 'timeline' | 'models' = 'dimensions';
  modelView: 'matrix' | 'chart' = 'matrix';
  experimentFilter: string = '';

  get hasData(): boolean {
    return this.history.length > 0 || this.currentResults.length > 0;
  }

  get experiments(): string[] {
    const ids = this.history
      .map((e) => e.experiment?.experiment_id)
      .filter((id): id is string => !!id);
    return [...new Set(ids)];
  }

  get filteredHistory(): ResearchHistoryEntry[] {
    if (!this.experimentFilter) return this.history;
    return this.history.filter((e) => e.experiment?.experiment_id === this.experimentFilter);
  }

  get allResults(): ResearchScoredResult[] {
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
    return [...this.filteredHistory]
      .reverse()
      .map((e) => ({
        label: e.source_title.length > 28 ? e.source_title.slice(0, 28) + '…' : e.source_title,
        ii: e.avg_ii_score,
        cf: e.avg_cf_score,
      }));
  }

  get modelComparison(): ModelPoint[] {
    const src = this.currentResults.length > 0
      ? this.currentResults
      : this.history[0]?.results ?? [];
    return src.map((r) => ({
      model: r.model.split('/').pop() ?? r.model,
      ii: r.overall_information_integrity_score,
      cf: r.overall_cultural_fidelity_score,
    }));
  }

  get modelMatrix(): ModelComparisonMatrix {
    const src = this.currentResults.length > 0
      ? this.currentResults
      : this.history[0]?.results ?? [];
    const models = src.map((r) => r.model.split('/').pop() ?? r.model);

    const makeRows = (group: 'ii' | 'cf'): DimensionRow[] =>
      Object.entries(CATEGORY_META)
        .filter(([, meta]) => meta.group === group)
        .map(([id, meta]) => ({
          id,
          name: meta.name,
          group,
          cells: src.map((r) => ({
            model: r.model.split('/').pop() ?? r.model,
            severity: r.dimension_scores[id]?.severity ?? 0,
            confidence: r.dimension_scores[id]?.confidence ?? 'Low',
            explanation: r.dimension_scores[id]?.explanation ?? '',
          })),
        }));

    return { models, iiRows: makeRows('ii'), cfRows: makeRows('cf') };
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

  get activeExperimentLabel(): string {
    if (!this.experimentFilter) return 'All experiments';
    const meta = this.history.find((e) => e.experiment?.experiment_id === this.experimentFilter)?.experiment;
    if (!meta) return this.experimentFilter;
    return meta.experiment_name
      ? `${this.experimentFilter} · ${meta.experiment_name}`
      : this.experimentFilter;
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
