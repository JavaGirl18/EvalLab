import { Component } from '@angular/core';
import { RunPanelComponent } from './run-panel/run-panel';
import { ResultsTableComponent } from './results-table/results-table';
import { EvalResponse } from './services/eval';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RunPanelComponent, ResultsTableComponent],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App {
  results: EvalResponse['results'] = [];

  onResultsReady(response: EvalResponse) {
    this.results = response.results;
  }
}
