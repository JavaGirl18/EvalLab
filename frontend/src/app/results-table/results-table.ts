import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ScoreResult } from '../services/eval';
import { EvalService } from '../services/eval';

@Component({
  selector: 'app-results-table',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './results-table.html',
  styles: '',
})
export class ResultsTableComponent {
  @Input() results: ScoreResult[] = [];
  @Input() loading = false;

  expanded: Record<string, boolean> = {};
  liked: Record<string, boolean> = {};
  readonly TRUNCATE_LENGTH = 300;
  readonly skeletonCards = Array(6);

  constructor(private evalService: EvalService) {}

  get grouped(): Record<string, ScoreResult[]> {
    return this.results.reduce((acc, r) => {
      acc[r.variant] = acc[r.variant] || [];
      acc[r.variant].push(r);
      return acc;
    }, {} as Record<string, ScoreResult[]>);
  }

  get variants(): string[] {
    return Object.keys(this.grouped);
  }

  scoreColor(score: number): string {
    if (score >= 8) return 'text-green-600';
    if (score >= 6) return 'text-yellow-600';
    return 'text-red-500';
  }

  getDimScore(result: ScoreResult, dim: string): number {
    return result[dim as keyof ScoreResult] as number;
  }

  cardKey(result: ScoreResult): string {
    return `${result.model}-${result.variant}`;
  }

  isExpanded(result: ScoreResult): boolean {
    return !!this.expanded[this.cardKey(result)];
  }

  isLiked(result: ScoreResult): boolean {
    return !!this.liked[this.cardKey(result)];
  }

  toggleExpand(result: ScoreResult): void {
    const key = this.cardKey(result);
    this.expanded = { ...this.expanded, [key]: !this.expanded[key] };
  }

  toggleLike(result: ScoreResult): void {
    const key = this.cardKey(result);
    const nowLiked = !this.liked[key];
    // Spread into a new object so Angular detects the change and re-renders.
    this.liked = { ...this.liked, [key]: nowLiked };

    if (nowLiked) {
      this.evalService.savePreference({
        model: result.model,
        task: result.task,
        variant: result.variant,
        user_prompt: result.user_prompt,
      }).subscribe();
    }
  }

  displayText(result: ScoreResult): string {
    if (this.isExpanded(result) || result.response_text.length <= this.TRUNCATE_LENGTH) {
      return result.response_text;
    }
    return result.response_text.slice(0, this.TRUNCATE_LENGTH) + '...';
  }

  scoreDims = ['usefulness', 'clarity', 'confidence', 'reliability'];
}
