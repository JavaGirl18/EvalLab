import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { BaseChartDirective } from 'ng2-charts';
import { Chart, ChartData, ChartOptions, registerables } from 'chart.js';
import { EvalService } from '../services/eval';

Chart.register(...registerables);

@Component({
  selector: 'app-insights-panel',
  standalone: true,
  imports: [CommonModule, BaseChartDirective],
  templateUrl: './insights-panel.html',
  styles: '',
})
export class InsightsPanelComponent implements OnInit {
  collapsed = true;
  models = ['gpt-4o', 'gpt-4o-mini', 'gpt-3.5-turbo'];
  summary: Record<string, number> = {};

  barData: ChartData<'bar'> = {
    labels: this.models,
    datasets: [
      {
        label: 'Thumbs Up',
        data: [],
        backgroundColor: ['#3b82f6', '#8b5cf6', '#10b981'],
        borderRadius: 6,
      },
    ],
  };

  barOptions: ChartOptions<'bar'> = {
    responsive: true,
    plugins: { legend: { display: false } },
    scales: { y: { beginAtZero: true, ticks: { stepSize: 1 } } },
  };

  constructor(private evalService: EvalService) {}

  ngOnInit() {
    this.loadSummary();
  }

  loadSummary() {
    this.evalService.getPreferencesSummary().subscribe((data) => {
      this.summary = data;
      this.barData = {
        ...this.barData,
        datasets: [
          {
            ...this.barData.datasets[0],
            data: this.models.map((m) => data[m] ?? 0),
          },
        ],
      };
    });
  }

  toggle() {
    this.collapsed = !this.collapsed;
    if (!this.collapsed) this.loadSummary();
  }

  totalVotes(): number {
    return Object.values(this.summary).reduce((a, b) => a + b, 0);
  }
}
