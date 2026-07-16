import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { NgIcon, provideIcons } from '@ng-icons/core';
import { lucideGavel } from '@ng-icons/lucide';
import { RunPanelComponent } from './run-panel/run-panel';
import { ResultsTableComponent } from './results-table/results-table';
import { ResearchPanelComponent } from './research-panel/research-panel';
import { ResearchResultsComponent } from './research-results/research-results';
import { ResearchTrendsComponent } from './research-trends/research-trends';
import { HumanReviewDashboardComponent } from './human-review-dashboard/human-review-dashboard';
import { ExternalPackagesComponent } from './external-packages/external-packages';
import {
  EvalResponse,
  ResearchEvalResponse,
  ResearchScoredResult,
  ResearchHistoryEntry,
  ExperimentMeta,
} from './services/eval';

const HISTORY_KEY = 'evallab_research_history';
const MAX_HISTORY = 10;

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    CommonModule,
    RunPanelComponent,
    ResultsTableComponent,
    ResearchPanelComponent,
    ResearchResultsComponent,
    ResearchTrendsComponent,
    HumanReviewDashboardComponent,
    ExternalPackagesComponent,
    NgIcon,
  ],
  providers: [provideIcons({ lucideGavel })],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App {
  mode: 'eval' | 'research' = 'research';
  researchView: 'results' | 'trends' | 'reviews' | 'import' = 'results';
  sidebarCollapsed = false;
  currentExperimentId = '';

  // Model Eval state
  results: EvalResponse['results'] = [];
  loading = false;

  // Research Eval state
  researchResults: ResearchScoredResult[] = [];
  researchLoading = false;
  researchHistory: ResearchHistoryEntry[] = this.loadHistory();
  pendingExperiment: ExperimentMeta | null = null;

  onLoadingChange(isLoading: boolean) {
    this.loading = isLoading;
  }

  onResultsReady(response: EvalResponse) {
    this.results = response.results;
    this.loading = false;
  }

  onResearchLoadingChange(isLoading: boolean) {
    this.researchLoading = isLoading;
  }

  onResearchResultsReady(response: ResearchEvalResponse) {
    this.researchResults = response.results;
    this.researchLoading = false;
    if (response.results.length > 0) {
      const r0 = response.results[0];
      const avgII = response.results.reduce((s, r) => s + r.overall_information_integrity_score, 0) / response.results.length;
      const avgCF = response.results.reduce((s, r) => s + r.overall_cultural_fidelity_score, 0) / response.results.length;
      this.researchHistory = [
        {
          timestamp: r0.timestamp,
          source_title: r0.source_title,
          item_id: r0.item_id,
          variant: r0.prompt_variant_name,
          temperature: r0.temperature,
          avg_ii_score: Math.round(avgII * 10) / 10,
          avg_cf_score: Math.round(avgCF * 10) / 10,
          results: response.results,
          ...(this.pendingExperiment ? { experiment: this.pendingExperiment } : {}),
        },
        ...this.researchHistory,
      ].slice(0, MAX_HISTORY);
      this.pendingExperiment = null;
      this.saveHistory(this.researchHistory);
    }
  }

  onHistorySelect(entry: ResearchHistoryEntry) {
    this.researchResults = entry.results;
  }

  onExperimentReady(exp: ExperimentMeta | null) {
    this.pendingExperiment = exp;
    if (exp?.experiment_id) {
      this.currentExperimentId = exp.experiment_id;
    }
  }

  onHistoryTagged({ timestamp, experiment }: { timestamp: string; experiment: ExperimentMeta }) {
    this.researchHistory = this.researchHistory.map((e) =>
      e.timestamp === timestamp ? { ...e, experiment } : e
    );
    this.saveHistory(this.researchHistory);
  }

  private loadHistory(): ResearchHistoryEntry[] {
    try {
      const raw = localStorage.getItem(HISTORY_KEY);
      return raw ? JSON.parse(raw) : [];
    } catch {
      return [];
    }
  }

  private saveHistory(history: ResearchHistoryEntry[]): void {
    try {
      localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
    } catch {
      // localStorage full or unavailable — silently skip
    }
  }
}
