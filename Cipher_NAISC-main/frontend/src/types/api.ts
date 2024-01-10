// API types for the Cipher_NAISC backend

export interface ApiIncident {
  id: string
  timestamp: string
  detections: unknown
  perception_summary: string
  recommended_action: string
  reasoning_confidence: number
  officer_action: string | null
  final_outcome: string | null
  feedback: string | null
  is_false_positive: boolean
  agent_reports: Record<string, Record<string, unknown>> | null
  created_at: string
  updated_at: string
}

export interface OfficerResponsePayload {
  officer_action: string
  final_outcome: string
  feedback: string
  is_false_positive: boolean
}

export interface AnalyticsData {
  total_incidents: number
  false_positive_count: number
  false_positive_rate: number
  responded_count: number
  incidents_per_day: { day: string; count: number }[]
  action_distribution: { recommended_action: string; count: number }[]
  learning?: {
    total_incidents: number
    responded: number
    recommendation_accuracy: number
  }
  ml_stats?: {
    accuracy: number
    samples_trained: number
    last_updated: string | null
    is_active: boolean
  }
  ml_adjusted_alerts?: number
  false_positives_prevented?: number
  most_common_threat?: string
}
