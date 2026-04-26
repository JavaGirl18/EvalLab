import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable, Subject, tap } from 'rxjs';
import { environment } from '../../environments/environment';

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

@Injectable({
  providedIn: 'root',
})
export class EvalService {
  private apiUrl = environment.apiUrl;

  // Emits whenever a preference is successfully saved so the insights
  // panel knows to reload its chart data in real time.
  preferenceAdded$ = new Subject<void>();

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
}
