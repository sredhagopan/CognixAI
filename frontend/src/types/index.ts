// ── Patient & Clinical Data ──────────────────────────────────────────────────

export type TrajectoryLabel = 'Improving' | 'Stable' | 'Deteriorating'
export type ConfidenceLevel = 'Low' | 'Moderate' | 'High'

export interface ShapFactor {
  feature: string
  display_name: string
  shap_value: number
  direction: 'increases' | 'decreases'
}

export interface ClinicalProfile {
  age: string
  gender: string
  disease: string
  medical_history: string
  stage: string
  bmi: string
  cholesterol: string
  heart_rate: string
  bp_systolic: string
  bp_diastolic: string
  cognitive_score: string
  cognitive_baseline: string
  cognitive_delta: string
  biomarker_score: string
  medication_dose: string
  medication_adherence: string
  sleep_hours: string
  steps_per_day: string
  stress_level: string
  mood_score: string
  lifestyle: string
  alcohol_use: string
  smoker: string
  support_system: string
  has_caregiver: string
  employment: string
  visit_index: string
  visit_date: string
}

export interface FeatureFactor {
  feature: string
  display_name: string
  shap_importance: number
  shap_direction: 'increases' | 'decreases'
  patient_value: number
  // continuous
  phenotype_mean?: number
  phenotype_median?: number
  phenotype_std?: number
  percentile?: number
  z_score?: number
  interpretation: string
  // binary
  phenotype_prevalence_pct?: number
}

export interface ReasoningData {
  n_peers: number
  overall_summary: string
  top_actionable_factors: FeatureFactor[]
  top_non_actionable_factors: FeatureFactor[]
  cautions: string[]
}

export interface ClinicalReasoning {
  major_concerns: string[]
  protective_factors: string[]
  actionable_priorities: string[]
  non_actionable_factors: string[]
  peer_summary: string
  confidence_level: ConfidenceLevel
  confidence_statement: string
  recommended_follow_up_questions: string[]
}

export interface FeatureTrend {
  direction: 'improving' | 'worsening' | 'stable'
  magnitude: 'slight' | 'moderate' | 'notable'
  n_visits: number
  recent: number | null
  first: number | null
}

export interface PhenotypePeerStats {
  Phenotype_Simplified?: string
  N_Patients?: number
  Pct_of_Population?: number
  'Improving_%'?: number
  'Stable_%'?: number
  'Deteriorating_%'?: number
}

export interface RagInfo {
  phenotype?: string
  domain?: string
  dominant_feature?: string
  n_patients?: number
  prevalence_pct?: number
  improving_pct?: number
  stable_pct?: number
  deteriorating_pct?: number
  characteristics?: string
  recommendation?: string
  evidence_context?: string
  chatbot_responses?: Record<string, string>
}

export interface CognitiveScorePoint {
  date: string
  score: number
  visit_index: number | null
}

export interface PatientData {
  patient_id: string
  phenotype: string
  phenotype_summary: string
  predicted_score: number | null
  predicted_label: TrajectoryLabel
  predicted_change: number | null
  n_patients: number | string
  prevalence_pct: number | string
  improving_pct: number
  stable_pct: number
  deteriorating_pct: number
  characteristics: string
  recommendation: string
  domain: string
  dominant_feature: string
  shap_factors: ShapFactor[]
  clinical_profile: ClinicalProfile
  reasoning: ReasoningData
  clinical_reasoning: ClinicalReasoning
  feature_trends: Record<string, FeatureTrend>
  peer_stats: PhenotypePeerStats
  rag_info: RagInfo
  cognitive_score_history: CognitiveScorePoint[]
}

// ── Chat ─────────────────────────────────────────────────────────────────────

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
  simResult?: SimulationResult
  isStreaming?: boolean
}

// ── Simulation ────────────────────────────────────────────────────────────────

export interface ShapShift {
  feature: string
  display_name: string
  before: number
  after: number
  shift: number
}

export interface IndividualImpact {
  feature: string
  value: number
  delta: number
}

export interface SimulationResult {
  patient_id: string
  changes_applied: Record<string, number>
  original_values: Record<string, number>
  before_score: number
  after_score: number
  delta: number
  before_label: TrajectoryLabel
  after_label: TrajectoryLabel
  label_changed: boolean
  shap_shifts: ShapShift[]
  mode?: 'best_single' | 'best_combination' | 'specific'
  found?: boolean
  best_feature?: string
  best_value?: number
  individual_impacts?: IndividualImpact[]
}

// ── Feature Importance ────────────────────────────────────────────────────────

export interface FeatureImportanceRow {
  feature: string       // lowercase — matches CSV column name
  mean_abs_shap: number // lowercase — matches CSV column name
}

// ── Simulation Metadata ───────────────────────────────────────────────────────

export interface ClinicalRange {
  label: string
  lower: number
  upper: number
}

export interface SimulationMeta {
  feature_aliases: Record<string, string>
  realistic_targets: Record<string, number[]>
  clinical_ranges: Record<string, [string, number, number][]>
  feature_cascades: Record<string, string[]>
}

// ── API Response wrappers ─────────────────────────────────────────────────────

export interface ApiError {
  error: string
  trace?: string
}
