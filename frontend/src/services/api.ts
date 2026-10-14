import {
  HealthResponse,
  ModelInfoResponse,
  PredictionResponse,
  HistoryListResponse,
  HistoryDetail,
  SystemInfoResponse,
} from '../types';

const BASE_URL = import.meta.env.VITE_API_URL || '/api/v1';

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let errorDetail = 'API request failed';
    try {
      const errJson = await res.json();
      errorDetail = errJson.detail || errJson.message || errorDetail;
    } catch {
      errorDetail = `HTTP ${res.status}: ${res.statusText}`;
    }
    throw new Error(errorDetail);
  }
  return res.json();
}

export const api = {
  async getHealth(): Promise<HealthResponse> {
    const res = await fetch(`${BASE_URL}/health`);
    return handleResponse<HealthResponse>(res);
  },

  async getModelInfo(): Promise<ModelInfoResponse> {
    const res = await fetch(`${BASE_URL}/model-info`);
    return handleResponse<ModelInfoResponse>(res);
  },

  async getSystemInfo(): Promise<SystemInfoResponse> {
    const res = await fetch(`${BASE_URL}/system-info`);
    return handleResponse<SystemInfoResponse>(res);
  },

  async predict(file: File): Promise<PredictionResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const res = await fetch(`${BASE_URL}/predict`, {
      method: 'POST',
      body: formData,
    });
    return handleResponse<PredictionResponse>(res);
  },

  async getHistory(params?: {
    limit?: number;
    offset?: number;
    search?: string;
    species?: string;
    grade?: number;
    is_ood?: boolean;
  }): Promise<HistoryListResponse> {
    const query = new URLSearchParams();
    if (params?.limit !== undefined) query.set('limit', String(params.limit));
    if (params?.offset !== undefined) query.set('offset', String(params.offset));
    if (params?.search) query.set('search', params.search);
    if (params?.species) query.set('species', params.species);
    if (params?.grade !== undefined) query.set('grade', String(params.grade));
    if (params?.is_ood !== undefined) query.set('is_ood', String(params.is_ood));

    const qs = query.toString();
    const url = `${BASE_URL}/history${qs ? `?${qs}` : ''}`;
    const res = await fetch(url);
    return handleResponse<HistoryListResponse>(res);
  },

  async getHistoryDetail(id: string): Promise<HistoryDetail> {
    const res = await fetch(`${BASE_URL}/history/${encodeURIComponent(id)}`);
    return handleResponse<HistoryDetail>(res);
  },

  async deleteHistoryItem(id: string): Promise<{ status: string; message: string }> {
    const res = await fetch(`${BASE_URL}/history/${encodeURIComponent(id)}`, {
      method: 'DELETE',
    });
    return handleResponse<{ status: string; message: string }>(res);
  },

  async clearHistory(): Promise<{ status: string; message: string }> {
    const res = await fetch(`${BASE_URL}/history?confirm=true`, {
      method: 'DELETE',
    });
    return handleResponse<{ status: string; message: string }>(res);
  },
};
