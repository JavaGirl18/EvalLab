import { Component, Input } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ScoreResult } from '../services/eval';

@Component({
  selector: 'app-results-table',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './results-table.html',
  styles: '',
})
export class ResultsTableComponent {
  @Input() results: ScoreResult[] = [];

  // Groups results by variant so we can render zero_shot and role_prompted side by side.
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

  scoreDims = ['usefulness', 'clarity', 'confidence', 'reliability'];
}
