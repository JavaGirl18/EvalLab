import { Component, Input, OnInit, OnChanges, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { BaseChartDirective } from 'ng2-charts';
import { Chart, ChartData, ChartOptions, registerables } from 'chart.js';
import { EvalService, ScoreResult } from '../services/eval';
import { Subscription } from 'rxjs';

Chart.register(...registerables);

const MODEL_COLORS: Record<string, string> = {
  'gpt-4o':        'rgba(59, 130, 246, 0.6)',
  'gpt-4o-mini':   'rgba(139, 92, 246, 0.6)',
  'gpt-3.5-turbo': 'rgba(16, 185, 129, 0.6)',
};

@Component({
  selector: 'app-insights-panel',
  standalone: true,
  imports: [CommonModule, BaseChartDirective],
  templateUrl: './insights-panel.html',
  styles: '',
})
export class InsightsPanelComponent implements OnInit, OnChanges, OnDestroy {
  @Input() results: ScoreResult[] = [];

  collapsed = true;
  models = ['gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo'];
  dims = ['usefulness', 'clarity', 'confidence', 'reliability'];
  summary: Record<string, number> = {};
  private sub = new Subscription();

  barData: ChartData<'bar'> = {
    labels: this.models,
    datasets: [{
      label: 'Thumbs Up',
      data: [],
      backgroundColor: ['#3b82f6', '#8b5cf6', '#10b981'],
      borderRadius: 6,
    }],
  };

  barOptions: ChartOptions<'bar'> = {
    responsive: true,
    plugins: { legend: { display: false } },
    scales: { y: { beginAtZero: true, ticks: { stepSize: 1 } } },
  };

  radarData: ChartData<'radar'> = { labels: [], datasets: [] };

  radarOptions: ChartOptions<'radar'> = {
    responsive: true,
    scales: {
      r: { beginAtZero: true, min: 0, max: 10, ticks: { stepSize: 2 } },
    },
    plugins: { legend: { position: 'bottom', labels: { boxWidth: 12, font: { size: 11 } } } },
  };

  constructor(private evalService: EvalService) {}

  ngOnInit() {
    this.loadSummary();
    this.sub.add(
      this.evalService.preferenceAdded$.subscribe(() => this.loadSummary())
    );
  }

  ngOnChanges() {
    this.buildRadar();
  }

  ngOnDestroy() {
    this.sub.unsubscribe();
  }

  loadSummary() {
    this.evalService.getPreferencesSummary().subscribe((data) => {
      this.summary = data;
      this.barData = {
        ...this.barData,
        datasets: [{
          ...this.barData.datasets[0],
          data: this.models.map((m) => data[m] || 0),
        }],
      };
    });
  }

  buildRadar() {
    if (!this.results.length) return;

    // Compute average score per dimension per model across all results.
    const datasets = this.models.map((model) => {
      const modelResults = this.results.filter((r) => r.model === model);
      const avg = (dim: string) => {
        if (!modelResults.length) return 0;
        return modelResults.reduce((sum, r) => sum + (r[dim as keyof ScoreResult] as number), 0) / modelResults.length;
      };

      return {
        label: model,
        data: this.dims.map((d) => avg(d)),
        backgroundColor: MODEL_COLORS[model] ?? 'rgba(100,100,100,0.4)',
        borderColor: MODEL_COLORS[model]?.replace('0.6', '1') ?? '#666',
        pointBackgroundColor: MODEL_COLORS[model]?.replace('0.6', '1') ?? '#666',
        borderWidth: 2,
      };
    });

    this.radarData = {
      labels: this.dims.map((d) => d.charAt(0).toUpperCase() + d.slice(1)),
      datasets,
    };
  }

  toggle() {
    this.collapsed = !this.collapsed;
    if (!this.collapsed) {
      this.loadSummary();
      this.buildRadar();
    }
  }

  totalVotes(): number {
    return Object.values(this.summary).reduce((a, b) => a + b, 0);
  }
}
