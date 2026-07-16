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

export interface DocumentCharacteristics {
  benchmark_category: string;
  cultural_significance: 'low' | 'medium' | 'high';
  primary_community: string | null;
  geographic_context: string;
  domain: 'health' | 'politics' | 'culture' | 'legal' | 'economy' | 'education' | 'community' | 'environment' | 'technology' | 'other';
  contains_multiple_perspectives: boolean;
  contains_uncertainty: boolean;
  contains_disputed_claims: boolean;
  contains_direct_quotes: boolean;
  contains_statistics: boolean;
  transformation_risk: 'low' | 'medium' | 'high';
}

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
  characteristics?: DocumentCharacteristics;
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

export interface ExperimentStatus {
  phase: string;           // planned | in_progress | completed | paused
  started_at: string | null;
  completed_at: string | null;
  created_by: string;
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
  status?: ExperimentStatus;
}

export interface ExperimentFileSummary {
  experiment_id: string;
  experiment_name: string;
  research_phase: string;
  status_phase: string;
}

export interface ExperimentFileConfig {
  experiment_id: string;
  experiment_name: string;
  research_phase: string;
  research_objective: string;
  research_question: string;
  hypothesis: string;
  transformation_task?: { id: string; name: string; instructions: string };
  prompt_variant: string;
  subject_models: string[];
  judge_model?: string;
  benchmark_corpus?: string;
  batch_config?: string;
  status?: {
    phase: string;
    started_at: string | null;
    completed_at: string | null;
    created_by: string;
  };
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

// ── Batch Eval types ──────────────────────────────────────────────────────────

export interface BatchRequest {
  experiment_id: string;
  item_ids: string[];
  prompt_variants: string[];
  models: string[];
  temperature: number;
  task_instructions?: string;
  experiment_meta?: ExperimentMeta;
  max_concurrency?: number;
  retry_limit?: number;
  resume_existing?: boolean;
  max_runs?: number;
}

export interface BatchRunItem {
  run_key: string;
  item_id: string;
  prompt_variant: string;
  model: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'skipped' | 'cancelled';
  attempt: number;
  error: string | null;
  started_at: string | null;
  completed_at: string | null;
}

export interface BatchStatus {
  batch_id: string;
  experiment_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  total: number;
  completed: number;
  failed: number;
  skipped: number;
  pending: number;
  warnings: string[];
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  items: BatchRunItem[];
}

export interface BatchSummary {
  batch_id: string;
  experiment_id: string;
  status: string;
  total: number;
  completed: number;
  failed: number;
  pending: number;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
}

export interface BatchResults {
  batch_id: string;
  experiment_id: string;
  experiment_meta?: ExperimentMeta;
  results: ResearchScoredResult[];
  summary: {
    expected: number;
    completed: number;
    failed: number;
    skipped: number;
    pending: number;
    avg_ii_by_model: Record<string, number>;
    avg_cf_by_model: Record<string, number>;
    top_failure_categories: [string, number][];
  };
}

// ── External Package types ────────────────────────────────────────────────────

export interface ExternalPackageFile {
  name: string;
  size: number;
  type: string;
  allowed: boolean;
}

export interface ExternalPackage {
  pkg_id: string;
  status: 'imported' | 'reviewed' | 'approved' | 'rejected';
  source_label: string;
  submitted_by: string;
  received_at: string;
  reviewed_at: string | null;
  approved_at: string | null;
  rejected_at: string | null;
  rejection_reason: string | null;
  file_manifest: ExternalPackageFile[];
  detected_meta: Record<string, any>;
  mapped_meta: Record<string, any>;
  notes: string;
  evaluation_id: string | null;
  storage_dir: string;
}

// ── Human Review types ────────────────────────────────────────────────────────

export interface GenerateReviewsRequest {
  batch_id: string;
  review_round?: number;
  blinded?: boolean;
  max_source_chars?: number | null;
  severity_threshold?: number | null;
  confidence_threshold?: string | null;
  failure_count_threshold?: number | null;
  random_pct?: number | null;
  manual_run_keys?: string[] | null;
  representative_sample?: boolean;
}

export interface HumanReviewSummary {
  review_id: string;
  experiment_id: string;
  batch_id: string;
  run_key: string;
  dataset_item_id: string;
  prompt_variant: string;
  review_round: number;
  selection_reasons: string[];
  review_status: 'pending' | 'exported' | 'completed' | 'archived';
  blinded: boolean;
  created_at: string;
  exported_at: string | null;
  completed_at: string | null;
  // From packet_snapshot via json_extract
  source_title?: string | null;
  benchmark_category?: string | null;
  cultural_significance?: string | null;
}

export interface HumanReviewResponse {
  response_id: string;
  review_id: string;
  reviewer_id: string;
  agreement_with_judge: string;
  disagreement_types: string[];
  comments: string;
  missed_failures: string;
  incorrectly_flagged: string;
  preserved_meaning: string;
  cultural_context_preserved: string;
  additional_comments: string;
  reviewed_at: string;
  imported_at: string;
  import_source: string;
}

export interface EmergingPatterns {
  high_disagreement_categories: Array<{
    category_id: string;
    agree_pct: number;
    yes: number;
    partially: number;
    no: number;
    total: number;
  }>;
  disputed_articles: Array<{
    item_id: string;
    title: string;
    partial_or_no: number;
    yes: number;
    total: number;
    dispute_rate: number;
  }>;
  reviewer_consistency: Array<{
    reviewer_id: string;
    agree: number;
    total: number;
    agree_pct: number;
  }>;
  severity_calibration: Array<{
    severity: string;
    yes: number;
    partial_or_no: number;
    total: number;
    agree_pct: number;
  }>;
  top_disagreement_types: [string, number][];
  total_completed: number;
}

export interface JudgeFinding {
  category_id: string;
  plain_label: string;
  severity: number;
  severity_label: string;
  confidence: string;
  explanation: string;
}

export interface PacketSnapshot {
  source_title: string;
  source_publisher: string;
  source_published_date: string;
  source_text: string;
  benchmark_category: string;
  cultural_significance: string;
  transformation_task: string;
  response_label: string;
  response_text: string;
  judge_findings: JudgeFinding[];
  judge_summary: string;
  judge_ii_score: number;
  judge_cf_score: number;
  taxonomy_version: string;
  rubric_version: string;
  snapshot_created_at: string;
}

export interface HumanReviewDetail extends HumanReviewSummary {
  packet_snapshot: PacketSnapshot;
}

export interface HumanReviewCounts {
  pending: number;
  exported: number;
  completed: number;
  archived: number;
  total: number;
}

export interface AgreementBucket {
  yes: number;
  yes_pct: number;
  partially: number;
  partially_pct: number;
  no: number;
  no_pct: number;
  unable_to_determine: number;
  unable_pct: number;
}

export interface HumanReviewStats {
  total_assigned: number;
  total_completed: number;
  total_responses: number;
  completion_rate: number;
  agreement: AgreementBucket;
  by_category: Record<string, Record<string, number>>;
  by_severity_level: Record<string, Record<string, number>>;
  by_prompt_variant: Record<string, Record<string, number>>;
  by_document: Record<string, Record<string, number>>;
  by_benchmark_category: Record<string, Record<string, number>>;
  by_cultural_significance: Record<string, Record<string, number>>;
  by_reviewer: Record<string, Record<string, number>>;
  top_disagreement_types: [string, number][];
}

export interface ImportResult {
  imported: number;
  updated: number;
  errors: Array<{ row: number; review_id: string; reason: string }>;
  unknown_ids: string[];
  duplicate_skipped: number;
}

export interface EvalRecord {
  record_format_version: string;
  record_exported_at: string;
  run_key: string;
  experiment_id: string;
  batch_id: string;
  item_id: string;
  source_title: string;
  source_publisher: string;
  source_published_date: string;
  source_url: string;
  benchmark_category: string;
  cultural_significance: string;
  article_type: string;
  subject_model: string;
  prompt_variant: string;
  temperature: number;
  response_text: string;
  judge_model: string;
  overall_ii_score: number;
  overall_cf_score: number;
  executive_summary: string;
  most_significant_failures: string[];
  suggested_improvements: string;
  dimension_scores: Record<string, DimensionScore>;
  taxonomy_version: string;
  rubric_version: string;
  human_validation: {
    review_id: string;
    review_round: number;
    blinded: boolean;
    selection_reasons: string[];
    responses: HumanReviewResponse[];
  } | null;
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

  createBatch(request: BatchRequest): Observable<BatchStatus> {
    return this.http.post<BatchStatus>(`${this.apiUrl}/batch`, request);
  }

  getBatchStatus(batchId: string): Observable<BatchStatus> {
    return this.http.get<BatchStatus>(`${this.apiUrl}/batch/${batchId}`);
  }

  getBatchResults(batchId: string): Observable<BatchResults> {
    return this.http.get<BatchResults>(`${this.apiUrl}/batch/${batchId}/results`);
  }

  cancelBatch(batchId: string): Observable<{ batch_id: string; status: string }> {
    return this.http.post<{ batch_id: string; status: string }>(
      `${this.apiUrl}/batch/${batchId}/cancel`, {}
    );
  }

  listBatches(): Observable<BatchSummary[]> {
    return this.http.get<BatchSummary[]>(`${this.apiUrl}/batches`);
  }

  listExperiments(): Observable<ExperimentFileSummary[]> {
    return this.http.get<ExperimentFileSummary[]>(`${this.apiUrl}/experiments`);
  }

  getExperiment(experimentId: string): Observable<ExperimentFileConfig> {
    return this.http.get<ExperimentFileConfig>(`${this.apiUrl}/experiment/${experimentId}`);
  }

  updateExperiment(experimentId: string, meta: ExperimentMeta): Observable<ExperimentFileConfig> {
    return this.http.put<ExperimentFileConfig>(`${this.apiUrl}/experiment/${experimentId}`, meta);
  }

  // ── Human Review ─────────────────────────────────────────────────────────────

  generateHumanReviews(request: GenerateReviewsRequest): Observable<{ created: number; skipped_duplicates: number; review_ids: string[] }> {
    return this.http.post<{ created: number; skipped_duplicates: number; review_ids: string[] }>(
      `${this.apiUrl}/human-reviews/generate`, request
    );
  }

  listHumanReviews(params: { batch_id?: string; experiment_id?: string; status?: string; review_round?: number } = {}): Observable<HumanReviewSummary[]> {
    const q = new URLSearchParams();
    if (params.batch_id)      q.set('batch_id',      params.batch_id);
    if (params.experiment_id) q.set('experiment_id', params.experiment_id);
    if (params.status)        q.set('status',        params.status);
    if (params.review_round)  q.set('review_round',  String(params.review_round));
    const qs = q.toString() ? `?${q}` : '';
    return this.http.get<HumanReviewSummary[]>(`${this.apiUrl}/human-reviews${qs}`);
  }

  countHumanReviews(batchId: string): Observable<HumanReviewCounts> {
    return this.http.get<HumanReviewCounts>(`${this.apiUrl}/human-reviews/counts?batch_id=${encodeURIComponent(batchId)}`);
  }

  getHumanReviewSummary(params: { batch_id?: string; experiment_id?: string; review_round?: number } = {}): Observable<HumanReviewStats> {
    const q = new URLSearchParams();
    if (params.batch_id)      q.set('batch_id',      params.batch_id);
    if (params.experiment_id) q.set('experiment_id', params.experiment_id);
    if (params.review_round)  q.set('review_round',  String(params.review_round));
    return this.http.get<HumanReviewStats>(`${this.apiUrl}/human-reviews/summary?${q}`);
  }

  exportHumanReviewUrl(format: string, params: { review_id?: string; batch_id?: string; experiment_id?: string; review_round?: number } = {}): string {
    const q = new URLSearchParams({ format });
    if (params.review_id)     q.set('review_id',     params.review_id);
    if (params.batch_id)      q.set('batch_id',      params.batch_id);
    if (params.experiment_id) q.set('experiment_id', params.experiment_id);
    if (params.review_round)  q.set('review_round',  String(params.review_round));
    return `${this.apiUrl}/human-reviews/export?${q}`;
  }

  importHumanReviews(file: File, overwrite = false): Observable<ImportResult> {
    const formData = new FormData();
    formData.append('file', file, file.name);
    return this.http.post<ImportResult>(
      `${this.apiUrl}/human-reviews/import?overwrite=${overwrite}`, formData
    );
  }

  getHumanReviewDetail(reviewId: string): Observable<HumanReviewDetail> {
    return this.http.get<HumanReviewDetail>(`${this.apiUrl}/human-reviews/${reviewId}`);
  }

  getHumanReviewResponses(reviewId: string): Observable<HumanReviewResponse[]> {
    return this.http.get<HumanReviewResponse[]>(`${this.apiUrl}/human-reviews/${reviewId}/responses`);
  }

  getHumanReviewPatterns(params: { batch_id?: string; experiment_id?: string; review_round?: number } = {}): Observable<EmergingPatterns> {
    const q = new URLSearchParams();
    if (params.batch_id)      q.set('batch_id',      params.batch_id);
    if (params.experiment_id) q.set('experiment_id', params.experiment_id);
    if (params.review_round)  q.set('review_round',  String(params.review_round));
    return this.http.get<EmergingPatterns>(`${this.apiUrl}/human-reviews/patterns?${q}`);
  }

  // ── Evaluation Records ────────────────────────────────────────────────────────

  registerEvaluationRecord(runKey: string): Observable<{ evaluation_id: string; run_key: string; registered_at: string; already_existed: boolean }> {
    return this.http.post<{ evaluation_id: string; run_key: string; registered_at: string; already_existed: boolean }>(
      `${this.apiUrl}/evaluation-records`,
      { run_key: runKey }
    );
  }

  listEvaluationRecords(params: { experiment_id?: string; batch_id?: string; limit?: number } = {}): Observable<Array<{ evaluation_id: string; run_key: string; registered_at: string; first_exported_at: string | null; export_count: number }>> {
    const q = new URLSearchParams();
    if (params.experiment_id) q.set('experiment_id', params.experiment_id);
    if (params.batch_id)      q.set('batch_id',      params.batch_id);
    if (params.limit)         q.set('limit',         String(params.limit));
    return this.http.get<Array<{ evaluation_id: string; run_key: string; registered_at: string; first_exported_at: string | null; export_count: number }>>(
      `${this.apiUrl}/evaluation-records?${q}`
    );
  }

  getEvalRecord(evaluationId: string): Observable<EvalRecord> {
    return this.http.get<EvalRecord>(`${this.apiUrl}/evaluation-records/${encodeURIComponent(evaluationId)}`);
  }

  exportEvalRecordUrl(evaluationId: string): string {
    return `${this.apiUrl}/evaluation-records/${encodeURIComponent(evaluationId)}/html`;
  }

  exportEvalRecordByRunKey(runKey: string): void {
    this.registerEvaluationRecord(runKey).subscribe({
      next: (reg) => window.open(this.exportEvalRecordUrl(reg.evaluation_id), '_blank'),
    });
  }

  // ── External Packages ───────────────────────────────────────────────────────

  uploadExternalPackage(formData: FormData): Observable<ExternalPackage> {
    return this.http.post<ExternalPackage>(`${this.apiUrl}/external-packages/upload`, formData);
  }

  listExternalPackages(): Observable<ExternalPackage[]> {
    return this.http.get<ExternalPackage[]>(`${this.apiUrl}/external-packages`);
  }

  getExternalPackage(pkgId: string): Observable<ExternalPackage> {
    return this.http.get<ExternalPackage>(`${this.apiUrl}/external-packages/${pkgId}`);
  }

  updateExternalPackageMeta(pkgId: string, mappedMeta: Record<string, any>, notes: string): Observable<ExternalPackage> {
    return this.http.put<ExternalPackage>(
      `${this.apiUrl}/external-packages/${pkgId}/meta`,
      { mapped_meta: mappedMeta, notes }
    );
  }

  approveExternalPackage(pkgId: string): Observable<{ evaluation_id: string; pkg_id: string }> {
    return this.http.post<{ evaluation_id: string; pkg_id: string }>(
      `${this.apiUrl}/external-packages/${pkgId}/approve`, {}
    );
  }

  rejectExternalPackage(pkgId: string, reason: string): Observable<ExternalPackage> {
    return this.http.post<ExternalPackage>(
      `${this.apiUrl}/external-packages/${pkgId}/reject`,
      { reason }
    );
  }

  evaluateExternalPackage(pkgId: string, req: {
    source_file: string;
    transformation_file: string;
    source_title?: string;
    subject_label?: string;
    task_description?: string;
  }): Observable<any> {
    return this.http.post<any>(`${this.apiUrl}/external-packages/${pkgId}/evaluate`, req);
  }

  getExternalPackageResult(pkgId: string): Observable<any> {
    return this.http.get<any>(`${this.apiUrl}/external-packages/${pkgId}/evaluate`);
  }

  exportExternalPackageRecord(pkgId: string, format: 'html' | 'json' = 'html'): Observable<Blob> {
    const path = format === 'html'
      ? `${this.apiUrl}/external-packages/${pkgId}/export/html`
      : `${this.apiUrl}/external-packages/${pkgId}/export`;
    return this.http.get(path, { responseType: 'blob' });
  }
}
