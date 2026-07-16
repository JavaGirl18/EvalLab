import { Component, OnInit, OnDestroy, Input, Output, EventEmitter, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import {
  EvalService,
  DatasetItem,
  FetchArticleResult,
  ResearchEvalRequest,
  ResearchEvalResponse,
  ResearchHistoryEntry,
  ExperimentMeta,
  ExperimentStatus,
  ExperimentFileSummary,
  HumanReviewCounts,
  HumanReviewSummary,
  HumanReviewStats,
  ImportResult,
} from '../services/eval';

const STANDARD_VARIANTS = [
  { name: 'baseline',               label: 'Baseline',                desc: 'Summarize key facts: who, what, when, where, outcomes.' },
  { name: 'safety_first',           label: 'Safety First',            desc: 'Summarize facts; flag harmful or unverified claims.' },
  { name: 'neutral',                label: 'Neutral',                 desc: 'Neutral 2–3 paragraph summary. No interpretation.' },
  { name: 'faithful_transformation', label: 'Faithful Transformation', desc: 'Complete the task exactly as instructed while preserving accuracy, attribution, and cultural context.' },
];

const CUSTOM_ITEMS_KEY = 'evallab_custom_articles';
const EXPERIMENT_KEY  = 'evallab_experiment_config';

const PHASES = [
  'pilot',
  'framework_validation',
  'case_study',
  'framework_refinement',
  'final_validation'
];

const KNOWN_MODELS = ['gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo'];

type SourceMode = 'dataset' | 'url' | 'paste';

@Component({
  selector: 'app-research-panel',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './research-panel.html',
})
export class ResearchPanelComponent implements OnInit, OnDestroy {
  @Input() history: ResearchHistoryEntry[] = [];
  @Output() resultsReady    = new EventEmitter<ResearchEvalResponse>();
  @Output() historySelect   = new EventEmitter<ResearchHistoryEntry>();
  @Output() historyTagged   = new EventEmitter<{ timestamp: string; experiment: ExperimentMeta }>();
  @Output() loadingChange   = new EventEmitter<boolean>();
  @Output() experimentReady = new EventEmitter<ExperimentMeta | null>();

  // Source mode
  sourceMode: SourceMode = 'dataset';

  // Dataset mode state
  items: DatasetItem[] = [];
  customItems: DatasetItem[] = [];
  loadingDataset = false;
  datasetError = '';
  selectedItemId = '';

  // URL mode state
  urlInput = '';
  fetchLoading = false;
  fetchError = '';
  fetchedArticle: FetchArticleResult | null = null;

  // Paste mode state
  pastedTitle = '';
  pastedText = '';
  pastedPublisher = '';
  pastedDate = '';

  // Common
  readonly variants = STANDARD_VARIANTS;
  selectedVariant = 'baseline';
  selectedVariants: string[] = [];   // multi-variant batch selection
  taskInstructions = '';
  humanNotes = '';
  temperature = 1.0;

  // Eval
  loading = false;
  evalError = '';

  // Sidebar tab
  sidebarTab: 'experiment' | 'run' | 'review' = 'experiment';

  // Collapsible sections
  articlesExpanded = true;
  historyExpanded = true;
  experimentExpanded = true;
  experimentSaved = false;

  // Experiment config
  readonly phases      = PHASES;
  readonly knownModels = KNOWN_MODELS;
  experimentId        = '';
  experimentName      = '';
  experimentPhase     = '';
  experimentObjective = '';
  experimentQuestion  = '';
  experimentHypothesis = '';
  experimentSourceId  = '';
  experimentVariant   = '';
  experimentModels: string[] = [];
  experimentStatusPhase = 'planned';
  experimentCreatedBy   = 'Valencia Cooper';
  experimentStartedAt   = '';
  experimentCompletedAt = '';

  readonly statusPhases = ['planned', 'in_progress', 'completed', 'paused'];

  // Experiment file loader
  availableExperiments: ExperimentFileSummary[] = [];
  selectedExperimentFile = '';

  constructor(private evalService: EvalService, private cdr: ChangeDetectorRef) {}

  ngOnInit() {
    this.loadDataset();
    this.loadCustomItems();
    this.loadExperimentConfig();
    this.loadExperimentFiles();
    this.resumeRunningBatch();
  }

  private resumeRunningBatch() {
    this.evalService.listBatches().subscribe({
      next: (batches) => {
        const running = batches.find((b) => b.status === 'running');
        if (running) {
          this.batchId        = running.batch_id;
          if (!this.hrBatchId) this.hrBatchId = running.batch_id;
          this.batchTotal     = running.total;
          this.batchCompleted = running.completed;
          this.batchFailed    = running.failed ?? 0;
          this.batchRunStatus = 'running';
          this.startBatchPolling(running.batch_id);
          this.cdr.detectChanges();
          return;
        }
        // Restore last batch for this experiment (for Human Review panel access after navigation)
        if (this.experimentId && !this.batchId) {
          const expBatches = batches
            .filter((b) => b.experiment_id === this.experimentId)
            .sort((a, b) => (b.created_at ?? '').localeCompare(a.created_at ?? ''));
          if (expBatches.length > 0) {
            this.batchId = expBatches[0].batch_id;
            if (!this.hrBatchId) this.hrBatchId = this.batchId;
            this.cdr.detectChanges();
          }
        }
      },
      error: () => {},
    });
  }

  loadExperimentFiles() {
    this.evalService.listExperiments().subscribe({
      next: (exps) => { this.availableExperiments = exps; this.cdr.detectChanges(); },
      error: () => {},
    });
  }

  loadExperiment(experimentId: string) {
    if (!experimentId) return;
    this.evalService.getExperiment(experimentId).subscribe({
      next: (exp) => {
        this.experimentId          = exp.experiment_id ?? '';
        this.experimentName        = exp.experiment_name ?? '';
        this.experimentPhase       = (exp.research_phase ?? '').toLowerCase().replace(/\s+/g, '_');
        this.experimentObjective   = exp.research_objective ?? '';
        this.experimentQuestion    = exp.research_question ?? '';
        this.experimentHypothesis  = exp.hypothesis ?? '';
        const expVariants = (exp as any).prompt_variants;
        this.experimentVariant = exp.prompt_variant || expVariants?.[0] || '';
        if (this.experimentVariant) this.selectedVariant = this.experimentVariant;
        this.selectedVariants = expVariants?.length > 1 ? expVariants : (this.experimentVariant ? [this.experimentVariant] : []);
        if ((exp as any).temperature !== undefined) this.temperature = (exp as any).temperature;
        this.taskInstructions      = exp.transformation_task?.instructions ?? '';
        this.experimentModels      = exp.subject_models ?? [];
        this.experimentStatusPhase = exp.status?.phase ?? 'planned';
        this.experimentCreatedBy   = exp.status?.created_by ?? 'Valencia Cooper';
        this.experimentStartedAt   = exp.status?.started_at ?? '';
        this.experimentCompletedAt = exp.status?.completed_at ?? '';
        this.persistExperimentConfig();
        this.cdr.detectChanges();
      },
      error: () => {},
    });
  }

  loadDataset() {
    this.loadingDataset = true;
    this.datasetError = '';
    this.evalService.getDataset().subscribe({
      next: (items) => {
        this.items = items;
        this.loadingDataset = false;
        this.cdr.detectChanges();
      },
      error: () => {
        this.loadingDataset = false;
        this.datasetError = 'Could not load dataset — is the backend running?';
        this.cdr.detectChanges();
      },
    });
  }

  private loadCustomItems() {
    try {
      const raw = localStorage.getItem(CUSTOM_ITEMS_KEY);
      this.customItems = raw ? JSON.parse(raw) : [];
    } catch {
      this.customItems = [];
    }
  }

  private persistCustomItems() {
    try {
      localStorage.setItem(CUSTOM_ITEMS_KEY, JSON.stringify(this.customItems));
    } catch {}
  }

  private saveCustomItem(item: DatasetItem) {
    this.customItems = [item, ...this.customItems].slice(0, 20);
    this.persistCustomItems();
  }

  removeCustomItem(id: string, event: Event) {
    event.stopPropagation();
    this.customItems = this.customItems.filter((i) => i.id !== id);
    if (this.selectedItemId === id) this.selectedItemId = '';
    this.persistCustomItems();
  }

  // ── Experiment config ────────────────────────────────────────────────────────

  private loadExperimentConfig() {
    try {
      const raw = localStorage.getItem(EXPERIMENT_KEY);
      if (raw) {
        const cfg: ExperimentMeta = JSON.parse(raw);
        this.experimentId          = cfg.experiment_id      ?? '';
        this.experimentName        = cfg.experiment_name    ?? '';
        this.experimentPhase       = cfg.phase              ?? '';
        this.experimentObjective   = cfg.research_objective ?? '';
        this.experimentQuestion    = cfg.research_question  ?? '';
        this.experimentHypothesis  = cfg.hypothesis         ?? '';
        this.experimentSourceId    = cfg.source_id          ?? '';
        const cfgVariants = (cfg as any).prompt_variants;
        this.experimentVariant = cfg.prompt_variant || cfgVariants?.[0] || '';
        if (this.experimentVariant) this.selectedVariant = this.experimentVariant;
        this.selectedVariants = cfgVariants?.length > 1 ? cfgVariants : (this.experimentVariant ? [this.experimentVariant] : []);
        if ((cfg as any).temperature !== undefined) this.temperature = (cfg as any).temperature;
        this.taskInstructions      = cfg.task_instructions  ?? '';
        this.experimentModels      = cfg.models             ?? [];
        this.experimentStatusPhase = cfg.status?.phase        ?? 'planned';
        this.experimentCreatedBy   = cfg.status?.created_by   ?? 'Valencia Cooper';
        this.experimentStartedAt   = cfg.status?.started_at   ?? '';
        this.experimentCompletedAt = cfg.status?.completed_at ?? '';
      }
    } catch {}
  }

  saveExperimentConfig() {
    this.persistExperimentConfig();
    const id = this.experimentId.trim();
    if (id) {
      this.evalService.updateExperiment(id, this.currentExperiment).subscribe({ error: () => {} });
    }
    this.experimentSaved = true;
    setTimeout(() => { this.experimentSaved = false; }, 2000);
  }

  private persistExperimentConfig() {
    const meta = this.currentExperiment;
    try {
      localStorage.setItem(EXPERIMENT_KEY, JSON.stringify(meta));
    } catch {}
    this.experimentReady.emit(meta);
  }

  get currentExperiment(): ExperimentMeta {
    return {
      experiment_id:      this.experimentId.trim(),
      experiment_name:    this.experimentName.trim(),
      phase:              this.experimentPhase,
      research_objective: this.experimentObjective.trim(),
      research_question:  this.experimentQuestion.trim(),
      hypothesis:         this.experimentHypothesis.trim(),
      source_id:          this.experimentSourceId.trim(),
      prompt_variant:     this.experimentVariant,
      ...(this.taskInstructions.trim() ? { task_instructions: this.taskInstructions.trim() } : {}),
      models:             this.experimentModels,
      status: {
        phase:        this.experimentStatusPhase,
        started_at:   this.experimentStartedAt   || null,
        completed_at: this.experimentCompletedAt || null,
        created_by:   this.experimentCreatedBy.trim(),
      } as ExperimentStatus,
    };
  }

  get hasExperiment(): boolean {
    return !!this.experimentId.trim();
  }

  toggleModel(model: string) {
    if (this.experimentModels.includes(model)) {
      this.experimentModels = this.experimentModels.filter((m) => m !== model);
    } else {
      this.experimentModels = [...this.experimentModels, model];
    }
  }

  isModelSelected(model: string): boolean {
    return this.experimentModels.includes(model);
  }

  autofillFromEval() {
    if (this.selectedItemId) this.experimentSourceId = this.selectedItemId;
  }

  tagHistoryEntry(entry: ResearchHistoryEntry) {
    this.historyTagged.emit({ timestamp: entry.timestamp, experiment: this.currentExperiment });
  }

  switchMode(mode: SourceMode) {
    this.sourceMode = mode;
    this.evalError = '';
  }

  // ── Dataset mode ─────────────────────────────────────────────────────────────

  get selectedItem(): DatasetItem | null {
    return (
      this.items.find((i) => i.id === this.selectedItemId) ??
      this.customItems.find((i) => i.id === this.selectedItemId) ??
      null
    );
  }

  selectItem(id: string) {
    this.selectedItemId = this.selectedItemId === id ? '' : id;
    this.evalError = '';
    this.humanNotes = this.selectedItem?.human_notes ?? '';
  }

  get datasetItemReady(): boolean {
    return !!this.selectedItem?.source_text?.trim();
  }

  isCustomItem(id: string): boolean {
    return id.startsWith('custom_');
  }

  sourceTypeLabel(type: string): string {
    const labels: Record<string, string> = {
      control: 'Control',
      international: 'International',
      diaspora: 'Diaspora',
      black_owned: 'Black-Owned',
      local_african: 'Local African',
      custom: 'Custom',
    };
    return labels[type] ?? type;
  }

  // ── URL fetch ────────────────────────────────────────────────────────────────

  fetchArticle() {
    if (!this.urlInput.trim()) return;
    this.fetchLoading = true;
    this.fetchError = '';
    this.fetchedArticle = null;

    this.evalService.fetchArticle(this.urlInput.trim()).subscribe({
      next: (result) => {
        this.fetchedArticle = result;
        this.fetchLoading = false;
        this.cdr.detectChanges();
      },
      error: (err) => {
        this.fetchError = err.error?.detail || 'Could not extract article from that URL.';
        this.fetchLoading = false;
        this.cdr.detectChanges();
      },
    });
  }

  clearFetch() {
    this.fetchedArticle = null;
    this.urlInput = '';
    this.fetchError = '';
  }

  get urlArticleReady(): boolean {
    return !!this.fetchedArticle?.text?.trim();
  }

  get urlTextPreview(): string {
    const text = this.fetchedArticle?.text ?? '';
    return text.length > 300 ? text.slice(0, 300) + '…' : text;
  }

  // ── Paste mode ───────────────────────────────────────────────────────────────

  get pastedWordCount(): number {
    return this.pastedText.trim() ? this.pastedText.trim().split(/\s+/).length : 0;
  }

  get pasteArticleReady(): boolean {
    return !!this.pastedText.trim();
  }

  clearPaste() {
    this.pastedTitle = '';
    this.pastedText = '';
    this.pastedPublisher = '';
    this.pastedDate = '';
  }

  // ── Run eval ─────────────────────────────────────────────────────────────────

  get needsTaskInstructions(): boolean {
    return this.selectedVariant === 'faithful_transformation';
  }

  get canRun(): boolean {
    if (this.loading) return false;
    if (this.needsTaskInstructions && !this.taskInstructions.trim()) return false;
    if (this.sourceMode === 'dataset') return this.datasetItemReady;
    if (this.sourceMode === 'url') return this.urlArticleReady;
    return this.pasteArticleReady;
  }

  runEval() {
    if (!this.canRun) return;
    this.loading = true;
    this.evalError = '';
    this.loadingChange.emit(true);
    this.persistExperimentConfig();

    const common: Pick<ResearchEvalRequest, 'temperature' | 'prompt_variant' | 'task_instructions'> = {
      temperature: this.temperature,
      prompt_variant: this.selectedVariant,
      ...(this.needsTaskInstructions && this.taskInstructions.trim()
        ? { task_instructions: this.taskInstructions.trim() }
        : {}),
    };
    let request: ResearchEvalRequest;

    if (this.sourceMode === 'dataset') {
      const item = this.selectedItem;
      if (item && this.isCustomItem(item.id)) {
        // Custom (saved) article — route as inline eval
        request = {
          ...common,
          source_url: item.source_url || undefined,
          source_title: item.source_title,
          source_text: item.source_text,
          human_notes: this.humanNotes,
        };
      } else {
        request = { ...common, item_id: this.selectedItemId, human_notes: this.humanNotes };
      }
    } else if (this.sourceMode === 'url') {
      request = {
        ...common,
        source_url: this.urlInput.trim(),
        source_title: this.fetchedArticle!.title,
        source_text: this.fetchedArticle!.text,
        human_notes: this.humanNotes,
      };
    } else {
      request = {
        ...common,
        source_title: this.pastedTitle.trim() || 'Untitled',
        source_text: this.pastedText,
        human_notes: this.humanNotes,
      };
    }

    this.evalService.runResearchEval(request).subscribe({
      next: (response) => {
        this.resultsReady.emit(response);
        this.loading = false;
        this.loadingChange.emit(false);
        this.autoSaveArticle();
        this.cdr.detectChanges();
      },
      error: (err) => {
        this.evalError = err.error?.detail || 'Something went wrong. Is the backend running?';
        this.loading = false;
        this.loadingChange.emit(false);
        this.cdr.detectChanges();
      },
    });
  }

  private autoSaveArticle() {
    if (this.sourceMode === 'url' && this.fetchedArticle) {
      this.saveCustomItem({
        id: `custom_${Date.now()}`,
        article_type: 'news',
        source_type: 'custom',
        high_context: false,
        expected_failure_categories: [],
        source_title: this.fetchedArticle.title,
        source_url: this.urlInput.trim(),
        source_text: this.fetchedArticle.text,
        metadata: {
          publisher: this.fetchedArticle.publisher,
          published_date: this.fetchedArticle.published_date,
        },
        prompt_variants: [],
        benchmark_rationale: '',
        human_notes: '',
        human_override: false,
      });
    } else if (this.sourceMode === 'paste' && this.pastedText.trim()) {
      this.saveCustomItem({
        id: `custom_${Date.now()}`,
        article_type: 'news',
        source_type: 'custom',
        high_context: false,
        expected_failure_categories: [],
        source_title: this.pastedTitle.trim() || 'Untitled',
        source_url: '',
        source_text: this.pastedText,
        metadata: {
          publisher: this.pastedPublisher,
          published_date: this.pastedDate,
        },
        prompt_variants: [],
        benchmark_rationale: '',
        human_notes: '',
        human_override: false,
      });
    }
  }

  // ── Batch ────────────────────────────────────────────────────────────────────

  batchLoading = false;
  batchSubmitted = false;
  batchId = '';
  batchError = '';
  batchTotal = 0;
  batchCompleted = 0;
  batchFailed = 0;
  batchFailedItems: { item_id: string; error: string | null }[] = [];
  batchFailuresExpanded = false;
  batchRunStatus: 'idle' | 'running' | 'completed' | 'failed' = 'idle';
  private batchPollRef: ReturnType<typeof setInterval> | null = null;

  ngOnDestroy() {
    this.clearBatchPoll();
  }

  private startBatchPolling(batchId: string) {
    this.batchPollRef = setInterval(() => {
      this.evalService.getBatchStatus(batchId).subscribe({
        next: (s) => {
          this.batchTotal     = s.total;
          this.batchCompleted = s.completed;
          this.batchFailed    = s.failed ?? 0;
          this.batchFailedItems = (s.items ?? [])
            .filter((i) => i.status === 'failed')
            .map((i) => ({ item_id: i.item_id, error: i.error }));
          if (s.status === 'completed' || s.status === 'failed' || s.status === 'cancelled') {
            this.batchRunStatus = s.status === 'completed' ? 'completed' : 'failed';
            this.clearBatchPoll();
          } else {
            this.batchRunStatus = 'running';
          }
          this.cdr.detectChanges();
        },
        error: () => {},
      });
    }, 4000);
  }

  private clearBatchPoll() {
    if (this.batchPollRef !== null) {
      clearInterval(this.batchPollRef);
      this.batchPollRef = null;
    }
  }

  get canRunBatch(): boolean {
    return !this.batchLoading && this.hasExperiment && this.items.length > 0;
  }

  runBatch() {
    if (!this.canRunBatch) return;
    this.batchLoading  = true;
    this.batchSubmitted = false;
    this.batchError    = '';

    const itemIds = this.items
      .filter((i) => i.source_text?.trim())
      .map((i) => i.id);

    const request = {
      experiment_id:   this.experimentId.trim(),
      item_ids:        itemIds,
      prompt_variants: this.selectedVariants.length > 0 ? this.selectedVariants : [this.experimentVariant || this.selectedVariant],
      models:          this.experimentModels.length > 0 ? this.experimentModels : ['gpt-5.6-sol', 'gemini-3.1-flash-lite', 'gpt-4o'],
      temperature:     this.temperature,
      ...(this.taskInstructions.trim() ? { task_instructions: this.taskInstructions.trim() } : {}),
      experiment_meta: this.currentExperiment,
      max_concurrency: 3,
      retry_limit:     3,
      resume_existing: true,
    };

    this.evalService.createBatch(request).subscribe({
      next: (status) => {
        this.batchId        = status.batch_id;
        this.batchTotal     = status.total;
        this.batchCompleted = 0;
        this.batchRunStatus = 'running';
        this.batchLoading   = false;
        this.batchSubmitted  = true;
        this.startBatchPolling(status.batch_id);
        this.cdr.detectChanges();
      },
      error: (err) => {
        this.batchError   = err.error?.detail
          ? (Array.isArray(err.error.detail) ? JSON.stringify(err.error.detail) : String(err.error.detail))
          : 'Batch submission failed.';
        this.batchLoading = false;
        this.cdr.detectChanges();
      },
    });
  }

  // ── Human Review ─────────────────────────────────────────────────────────────

  hrExpanded            = false;
  hrLoading             = false;
  hrError               = '';
  hrBatchId             = '';
  hrFilterMode: 'experiment' | 'batch' = 'experiment';
  hrSeverityThreshold: number | null   = 2;
  hrRandomPct: number | null           = null;
  hrFailureCount: number | null        = null;
  hrReviewRound                        = 1;
  hrBlinded                            = true;
  hrRepresentativeSample               = false;
  hrCounts: HumanReviewCounts | null   = null;
  hrReviews: HumanReviewSummary[]      = [];
  hrStats: HumanReviewStats | null     = null;
  hrImporting                          = false;
  hrImportResult: ImportResult | null  = null;
  hrImportError                        = '';

  toggleHrSection() {
    this.hrExpanded = !this.hrExpanded;
    if (this.hrExpanded) {
      if (!this.hrBatchId && this.batchId) this.hrBatchId = this.batchId;
      if (this.experimentId && this.hrFilterMode === 'experiment') {
        this.loadHrData();
      } else if (this.hrBatchId) {
        this.loadHrData();
      }
    }
  }

  loadHrData() {
    const useExp = this.hrFilterMode === 'experiment' && !!this.experimentId;
    const params = useExp
      ? { experiment_id: this.experimentId, review_round: this.hrReviewRound }
      : { batch_id: this.hrBatchId,         review_round: this.hrReviewRound };

    if (!params.experiment_id && !params.batch_id) return;

    if (params.batch_id) {
      this.evalService.countHumanReviews(params.batch_id).subscribe({
        next: (c) => { this.hrCounts = c; this.cdr.detectChanges(); },
        error: () => {},
      });
    }
    this.evalService.listHumanReviews(params).subscribe({
      next: (r) => { this.hrReviews = r; this.cdr.detectChanges(); },
      error: () => {},
    });
  }

  loadHrCounts() {
    this.loadHrData();
  }

  get hrReviewsByArticle(): { itemId: string; reviews: HumanReviewSummary[] }[] {
    const groups = new Map<string, HumanReviewSummary[]>();
    for (const r of this.hrReviews) {
      if (!groups.has(r.dataset_item_id)) groups.set(r.dataset_item_id, []);
      groups.get(r.dataset_item_id)!.push(r);
    }
    return Array.from(groups.entries()).map(([itemId, reviews]) => ({ itemId, reviews }));
  }

  get canGenerateHr(): boolean {
    return !this.hrLoading && !!this.hrBatchId.trim() &&
      (this.hrSeverityThreshold !== null || this.hrRandomPct !== null ||
       this.hrFailureCount !== null || this.hrRepresentativeSample);
  }

  generateHumanReviews() {
    if (!this.canGenerateHr) return;
    this.hrLoading = true;
    this.hrError   = '';

    this.evalService.generateHumanReviews({
      batch_id:                this.hrBatchId.trim(),
      review_round:            this.hrReviewRound,
      blinded:                 this.hrBlinded,
      severity_threshold:      this.hrSeverityThreshold,
      random_pct:              this.hrRandomPct,
      failure_count_threshold: this.hrFailureCount,
      representative_sample:   this.hrRepresentativeSample,
    }).subscribe({
      next: (result) => {
        this.hrLoading = false;
        this.hrError   = result.created === 0 && result.skipped_duplicates === 0
          ? 'No eligible runs found for those rules.'
          : '';
        this.loadHrData();
        this.cdr.detectChanges();
      },
      error: (err) => {
        this.hrError   = err.error?.detail || 'Failed to generate reviews.';
        this.hrLoading = false;
        this.cdr.detectChanges();
      },
    });
  }

  exportHrHtml(reviewId: string) {
    const url = this.evalService.exportHumanReviewUrl('html', { review_id: reviewId });
    window.open(url, '_blank');
  }

  hrStatusColor(status: string): string {
    if (status === 'completed') return 'text-green-600';
    if (status === 'exported')  return 'text-blue-500';
    if (status === 'archived')  return 'text-gray-400';
    return 'text-amber-500';
  }

  hrStatusLabel(status: string): string {
    if (status === 'completed') return '✓';
    if (status === 'exported')  return '↑';
    if (status === 'archived')  return '—';
    return '·';
  }

  exportHrCsvTemplate() {
    if (!this.hrBatchId) return;
    const url = this.evalService.exportHumanReviewUrl('csv_template', {
      batch_id:     this.hrBatchId,
      review_round: this.hrReviewRound,
    });
    window.open(url, '_blank');
  }

  exportHrContextCsv() {
    if (!this.hrBatchId) return;
    const url = this.evalService.exportHumanReviewUrl('csv_context', {
      batch_id:     this.hrBatchId,
      review_round: this.hrReviewRound,
    });
    window.open(url, '_blank');
  }

  exportHrJson() {
    if (!this.hrBatchId) return;
    const url = this.evalService.exportHumanReviewUrl('json', {
      batch_id:     this.hrBatchId,
      review_round: this.hrReviewRound,
    });
    window.open(url, '_blank');
  }

  importHrResponses(event: Event) {
    const input = event.target as HTMLInputElement;
    if (!input.files?.length) return;
    const file = input.files[0];
    this.hrImporting    = true;
    this.hrImportResult = null;
    this.hrImportError  = '';

    this.evalService.importHumanReviews(file).subscribe({
      next: (result) => {
        this.hrImportResult = result;
        this.hrImporting    = false;
        this.loadHrCounts();
        this.cdr.detectChanges();
        input.value = '';
      },
      error: (err) => {
        this.hrImportError = err.error?.detail || 'Import failed.';
        this.hrImporting   = false;
        this.cdr.detectChanges();
        input.value = '';
      },
    });
  }

  loadHrStats() {
    if (!this.hrBatchId) return;
    this.evalService.getHumanReviewSummary({ batch_id: this.hrBatchId, review_round: this.hrReviewRound }).subscribe({
      next: (s) => { this.hrStats = s; this.cdr.detectChanges(); },
      error: () => {},
    });
  }

  hrAgreementPct(key: 'yes_pct' | 'partially_pct' | 'no_pct' | 'unable_pct'): string {
    if (!this.hrStats) return '—';
    return `${Math.round((this.hrStats.agreement[key] ?? 0) * 100)}%`;
  }

  // ── History ──────────────────────────────────────────────────────────────────

  formatTime(ts: string): string {
    if (!ts) return '';
    return new Date(ts).toLocaleString(undefined, {
      month: 'short', day: 'numeric',
      hour: '2-digit', minute: '2-digit',
    });
  }

  scoreColor(score: number): string {
    if (score >= 8) return 'text-green-600';
    if (score >= 6) return 'text-yellow-500';
    return 'text-red-500';
  }
}
