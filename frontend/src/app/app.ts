import { Component } from '@angular/core';
import { NgIcon, provideIcons } from '@ng-icons/core';
import { lucideGavel } from '@ng-icons/lucide';
import { RunPanelComponent } from './run-panel/run-panel';
import { ResultsTableComponent } from './results-table/results-table';
import { EvalResponse } from './services/eval';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RunPanelComponent, ResultsTableComponent, NgIcon],
  providers: [provideIcons({ lucideGavel })],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App {
  results: EvalResponse['results'] = [];
  loading = false;

  onLoadingChange(isLoading: boolean) {
    this.loading = isLoading;
  }

  onResultsReady(response: EvalResponse) {
    this.results = response.results;
    this.loading = false;
  }
}
