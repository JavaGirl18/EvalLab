import { Component, Output, EventEmitter } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { EvalService, EvalRequest, EvalResponse } from '../services/eval';

@Component({
  selector: 'app-run-panel',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './run-panel.html',
  styles: '',
})
export class RunPanelComponent {
  @Output() resultsReady = new EventEmitter<EvalResponse>();

  task = 'resume';
  userPrompt = '';
  temperature = 0.7;
  loading = false;
  error = '';

  tasks = [
    { slug: 'resume', label: 'Resume Advice' },
    { slug: 'tax', label: 'Tax Help' },
    { slug: 'career', label: 'Career Transition' },
    { slug: 'budgeting', label: 'Budgeting' },
  ];

  constructor(private evalService: EvalService) {}

  runEval() {
    if (!this.userPrompt.trim()) return;

    this.loading = true;
    this.error = '';

    const request: EvalRequest = {
      task: this.task,
      user_prompt: this.userPrompt,
      temperature: this.temperature,
    };

    this.evalService.runEval(request).subscribe({
      next: (response) => {
        this.resultsReady.emit(response);
        this.loading = false;
      },
      error: (err) => {
        this.error = err.error?.detail || 'Something went wrong. Is the backend running?';
        this.loading = false;
      },
    });
  }
}
