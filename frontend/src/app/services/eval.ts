import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface EvalRequest {
  task: string;
  user_prompt: string;
  temperature: number;
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
}

export interface EvalResponse {
  results: ScoreResult[];
  csv_path: string;
}

@Injectable({
  providedIn: 'root',
})
export class EvalService {
  private apiUrl = 'http://localhost:8000';

  constructor(private http: HttpClient) {}

  runEval(request: EvalRequest): Observable<EvalResponse> {
    return this.http.post<EvalResponse>(`${this.apiUrl}/eval`, request);
  }
}
