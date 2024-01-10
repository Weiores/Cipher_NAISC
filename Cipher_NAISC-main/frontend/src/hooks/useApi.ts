import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import axios from 'axios'
import type { ApiIncident, AnalyticsData, OfficerResponsePayload } from '@/types/api'

const API_BASE = import.meta.env.VITE_API_URL ?? 'http://localhost:8000'

const api = axios.create({ baseURL: API_BASE })

// ── Incidents ────────────────────────────────────────────────────────────────

export function useIncidents(limit = 50) {
  return useQuery<ApiIncident[]>({
    queryKey: ['incidents', limit],
    queryFn: () => api.get(`/incidents?limit=${limit}`).then(r => r.data),
    refetchInterval: 5000,
  })
}

export function useIncident(id: string) {
  return useQuery<ApiIncident>({
    queryKey: ['incident', id],
    queryFn: () => api.get(`/incident/${id}`).then(r => r.data),
    enabled: Boolean(id),
  })
}

export function useSubmitResponse(incidentId: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (payload: OfficerResponsePayload) =>
      api.post(`/incident/${incidentId}/response`, payload).then(r => r.data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['incidents'] })
      qc.invalidateQueries({ queryKey: ['incident', incidentId] })
      qc.invalidateQueries({ queryKey: ['analytics'] })
    },
  })
}

// ── Agent reports ────────────────────────────────────────────────────────────

export function useAgentReports(incidentId: string, enabled: boolean) {
  return useQuery<Record<string, unknown>>({
    queryKey: ['agent-reports', incidentId],
    queryFn: () => api.get(`/incident/${incidentId}/agent-reports`).then(r => r.data),
    enabled: enabled && Boolean(incidentId),
    retry: false,
  })
}

// ── Analytics ────────────────────────────────────────────────────────────────

export function useAnalytics() {
  return useQuery<AnalyticsData>({
    queryKey: ['analytics'],
    queryFn: () => api.get('/analytics').then(r => r.data),
    refetchInterval: 30_000,
  })
}

// ── ML / Feedback ─────────────────────────────────────────────────────────────

export interface MLAccuracyPoint {
  timestamp: string
  accuracy: number
  samples_seen: number
}

export interface MLStats {
  is_fitted: boolean
  samples_seen: number
  accuracy: number
  last_updated: string | null
  sklearn_available: boolean
  accuracy_history: MLAccuracyPoint[]
}

export interface FeedbackSummary {
  total: number
  confirmed: number
  false_alarm: number
  partial: number
  good_rec: number
  bad_rec: number
  false_positive_rate: number
  recommendation_approval_rate: number
}

export function useMLStats() {
  return useQuery<MLStats>({
    queryKey: ['ml-stats'],
    queryFn: () => api.get('/ml-stats').then(r => r.data),
    refetchInterval: 60_000,
  })
}

export function useFeedbackSummary() {
  return useQuery<FeedbackSummary>({
    queryKey: ['feedback-summary'],
    queryFn: () => api.get('/feedback-summary').then(r => r.data),
    refetchInterval: 30_000,
  })
}

export function useMLRetrain() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: () => api.post('/ml-retrain').then(r => r.data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['ml-stats'] })
    },
  })
}
