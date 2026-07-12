import { Component, OnInit, Input, Output, EventEmitter, ChangeDetectorRef } from '@angular/core';
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
export class ResearchPanelComponent implements OnInit {
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
  taskInstructions = '';
  humanNotes = '';
  temperature = 0.7;

  // Eval
  loading = false;
  evalError = '';

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

  constructor(private evalService: EvalService, private cdr: ChangeDetectorRef) {}

  ngOnInit() {
    this.loadDataset();
    this.loadCustomItems();
    this.loadExperimentConfig();
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
        this.experimentVariant     = cfg.prompt_variant      ?? '';
        this.taskInstructions      = cfg.task_instructions  ?? '';
        this.experimentModels      = cfg.models             ?? [];
      }
    } catch {}
  }

  saveExperimentConfig() {
    this.persistExperimentConfig();
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
    if (this.selectedVariant) this.experimentVariant = this.selectedVariant;
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
