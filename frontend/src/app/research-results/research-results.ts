import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ResearchScoredResult, DimensionScore } from '../services/eval';

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
  imports: [CommonModule],
  templateUrl: './research-results.html',
})
export class ResearchResultsComponent {
  @Input() results: ResearchScoredResult[] = [];
  @Input() loading = false;

  showTaxonomyKey = false;

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
}
