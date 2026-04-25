import { Component, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { EvalService, EvalRequest, EvalResponse } from '../services/eval';
import { InsightsPanelComponent } from '../insights-panel/insights-panel';

interface HistoryEntry {
  task: string;
  taskLabel: string;
  prompt: string;
  temperature: number;
}

@Component({
  selector: 'app-run-panel',
  standalone: true,
  imports: [CommonModule, FormsModule, InsightsPanelComponent],
  templateUrl: './run-panel.html',
  styles: '',
})
export class RunPanelComponent {
  @Output() resultsReady = new EventEmitter<EvalResponse>();
  @Output() loadingChange = new EventEmitter<boolean>();

  task = 'resume';
  userPrompt = '';
  temperature = 0.7;
  loading = false;
  error = '';
  history: HistoryEntry[] = [];

  tasks = [
    { slug: 'resume', label: 'Resume Advice' },
    { slug: 'tax', label: 'Tax Help' },
    { slug: 'career', label: 'Career Transition' },
    { slug: 'budgeting', label: 'Budgeting' },
  ];

  constructor(private evalService: EvalService) {}

  onTaskChange() {
    this.userPrompt = '';
  }

  loadFromHistory(entry: HistoryEntry) {
    this.task = entry.task;
    this.userPrompt = entry.prompt;
    this.temperature = entry.temperature;
  }

  runEval() {
    if (!this.userPrompt.trim()) return;

    this.loading = true;
    this.error = '';
    this.loadingChange.emit(true);

    const request: EvalRequest = {
      task: this.task,
      user_prompt: this.userPrompt,
      temperature: this.temperature,
    };

    // Save to history before running
    const taskLabel = this.tasks.find(t => t.slug === this.task)?.label ?? this.task;
    this.history.unshift({
      task: this.task,
      taskLabel,
      prompt: this.userPrompt,
      temperature: this.temperature,
    });

    this.evalService.runEval(request).subscribe({
      next: (response) => {
        this.resultsReady.emit(response);
        this.loading = false;
        this.loadingChange.emit(false);
      },
      error: (err) => {
        this.error = err.error?.detail || 'Something went wrong. Is the backend running?';
        this.loading = false;
        this.loadingChange.emit(false);
      },
    });
  }
}
