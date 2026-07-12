import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, Subject, of, tap } from 'rxjs';

export interface EvalRequest {
  task: string;
  user_prompt: string;
  temperature: number;
}

export interface BiasFlag {
  quote: string;
  category: string;
  explanation: string;
}

export interface ScoreResult {
  model: string;
  task: string;
  variant: string;
  user_prompt: string;
  response_text: string;
  temperature: number;
  usefulness: number;
  clarity: number;
  confidence: number;
  reliability: number;
  judge_reasoning: string;
  needs_human_review: boolean;
  bias_score: number;
  bias_flags: BiasFlag[];
  bias_assessment: string;
}

export interface EvalResponse {
  results: ScoreResult[];
  csv_path: string;
}

export interface PreferenceRequest {
  model: string;
  task: string;
  variant: string;
  user_prompt: string;
}

// ── Research Eval types ───────────────────────────────────────────────────────

export interface DatasetItem {
  id: string;
  article_type: string;
  source_type: string;
  high_context: boolean;
  expected_failure_categories: string[];
  source_title: string;
  source_url: string;
  source_text: string;
  metadata: Record<string, string>;
  prompt_variants: Array<{ name: string; prompt: string }>;
  benchmark_rationale: string;
  human_notes: string;
  human_override: boolean;
}

export interface FetchArticleResult {
  title: string;
  text: string;
  publisher: string;
  published_date: string;
}

export interface ResearchEvalRequest {
  // Dataset mode
  item_id?: string;
  // Inline mode (from URL)
  source_url?: string;
  source_title?: string;
  source_text?: string;
  human_notes?: string;
  // Common
  temperature: number;
  prompt_variant?: string;
  task_instructions?: string;
}

export interface DimensionScore {
  severity: number;       // 0–3
  confidence: string;     // Low | Medium | High
  explanation: string;
  source_evidence: string;
  output_evidence: string;
}

export interface ResearchScoredResult {
  model: string;
  subject_model_config: Record<string, string>;
  item_id: string;
  article_type: string;
  source_title: string;
  system_prompt: string;
  eval_prompt: string;
  prompt_variant_name: string;
  response_text: string;
  temperature: number;
  timestamp: string;
  judge_subject_model_config: Record<string, string>;
  dimension_scores: Record<string, DimensionScore>;
  overall_information_integrity_score: number;
  overall_cultural_fidelity_score: number;
  executive_summary: string;
  most_significant_failures: string[];
  suggested_improvements: string;
}

export interface ResearchEvalResponse {
  results: ResearchScoredResult[];
  item_id: string;
}

export interface ExperimentMeta {
  experiment_id: string;
  experiment_name: string;
  phase: string;
  research_objective: string;
  research_question: string;
  hypothesis: string;
  source_id: string;
  prompt_variant: string;
  task_instructions?: string;
  models: string[];
}

export interface ResearchHistoryEntry {
  timestamp: string;
  source_title: string;
  item_id: string;
  variant: string;
  temperature: number;
  avg_ii_score: number;
  avg_cf_score: number;
  results: ResearchScoredResult[];
  experiment?: ExperimentMeta;
}

@Injectable({
  providedIn: 'root',
})
export class EvalService {
  private apiUrl = window.location.hostname === 'localhost'
    ? 'http://localhost:8000'
    : 'https://web-production-4a821.up.railway.app';

  // Emits whenever a preference is successfully saved so the insights
  // panel knows to reload its chart data in real time.
  preferenceAdded$ = new Subject<void>();
  private datasetCache: DatasetItem[] | null = null;

  constructor(private http: HttpClient) {}

  runEval(request: EvalRequest): Observable<EvalResponse> {
    return this.http.post<EvalResponse>(`${this.apiUrl}/eval`, request);
  }

  savePreference(request: PreferenceRequest): Observable<{ status: string }> {
    return this.http.post<{ status: string }>(`${this.apiUrl}/preference`, request).pipe(
      tap(() => this.preferenceAdded$.next())
    );
  }

  getPreferencesSummary(): Observable<Record<string, number>> {
    return this.http.get<Record<string, number>>(`${this.apiUrl}/preferences/summary`);
  }

  getDataset(): Observable<DatasetItem[]> {
    if (this.datasetCache) return of(this.datasetCache);
    return this.http.get<DatasetItem[]>(`${this.apiUrl}/dataset`).pipe(
      tap((items) => { this.datasetCache = items; })
    );
  }

  invalidateDatasetCache() {
    this.datasetCache = null;
  }

  fetchArticle(url: string): Observable<FetchArticleResult> {
    return this.http.post<FetchArticleResult>(`${this.apiUrl}/fetch-article`, { url });
  }

  runResearchEval(request: ResearchEvalRequest): Observable<ResearchEvalResponse> {
    return this.http.post<ResearchEvalResponse>(`${this.apiUrl}/research-eval`, request);
  }
}
