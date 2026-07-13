import type { TrajectoryLabel, ConfidenceLevel } from '../types'

// ── Trajectory helpers ────────────────────────────────────────────────────────

export function trajectoryColor(label: TrajectoryLabel | string): string {
  switch (label) {
    case 'Improving':     return 'text-emerald-600'
    case 'Stable':        return 'text-amber-600'
    case 'Deteriorating': return 'text-rose-600'
    default:              return 'text-slate-600'
  }
}

export function trajectoryBg(label: TrajectoryLabel | string): string {
  switch (label) {
    case 'Improving':     return 'bg-[#DFF6DD] text-[#107C10] ring-1 ring-[#C8E6C9]'
    case 'Stable':        return 'bg-[#FFF4CE] text-[#7A6200] ring-1 ring-[#F7E5A0]'
    case 'Deteriorating': return 'bg-[#FDE7E9] text-[#D13438] ring-1 ring-[#F4B8BA]'
    default:              return 'bg-[#F3F2F1] text-[#605E5C] ring-1 ring-[#EDEBE9]'
  }
}

export function trajectoryIcon(label: TrajectoryLabel | string): string {
  switch (label) {
    case 'Improving':     return '↑'
    case 'Stable':        return '→'
    case 'Deteriorating': return '↓'
    default:              return '–'
  }
}

export function trajectoryFill(label: TrajectoryLabel | string): string {
  switch (label) {
    case 'Improving':     return '#107C10'
    case 'Stable':        return '#C7A006'
    case 'Deteriorating': return '#D13438'
    default:              return '#A19F9D'
  }
}

// ── Confidence helpers ────────────────────────────────────────────────────────

export function confidenceColor(level: ConfidenceLevel | string): string {
  switch (level) {
    case 'High':     return 'text-emerald-600'
    case 'Moderate': return 'text-amber-600'
    case 'Low':      return 'text-rose-600'
    default:         return 'text-slate-600'
  }
}

export function confidenceBg(level: ConfidenceLevel | string): string {
  switch (level) {
    case 'High':     return 'bg-emerald-50 text-emerald-700 ring-1 ring-emerald-200'
    case 'Moderate': return 'bg-amber-50 text-amber-700 ring-1 ring-amber-200'
    case 'Low':      return 'bg-rose-50 text-rose-700 ring-1 ring-rose-200'
    default:         return 'bg-slate-100 text-slate-600'
  }
}

// ── SHAP direction helpers ────────────────────────────────────────────────────

export function shapDirectionColor(direction: string): string {
  return direction === 'increases' ? 'text-emerald-600' : 'text-rose-600'
}

export function shapDirectionIcon(direction: string): string {
  return direction === 'increases' ? '▲' : '▼'
}

export function shapBarColor(direction: string): string {
  return direction === 'increases' ? '#10b981' : '#f43f5e'
}

// ── Trend helpers ─────────────────────────────────────────────────────────────

export function trendIcon(direction: string): string {
  switch (direction) {
    case 'improving': return '↗'
    case 'worsening': return '↘'
    default:          return '→'
  }
}

export function trendColor(direction: string): string {
  switch (direction) {
    case 'improving': return 'text-[#107C10]'
    case 'worsening': return 'text-[#D13438]'
    default:          return 'text-[#A19F9D]'
  }
}

// ── Formatting ────────────────────────────────────────────────────────────────

export function fmt(val: string | number | null | undefined, decimals = 1): string {
  if (val == null || val === 'N/A') return 'N/A'
  const n = typeof val === 'string' ? parseFloat(val) : val
  if (isNaN(n)) return String(val)
  return n.toFixed(decimals)
}

export function fmtScore(val: number | null | undefined): string {
  if (val == null) return 'N/A'
  return val.toFixed(1)
}

export function fmtDelta(delta: number | null | undefined): string {
  if (delta == null) return 'N/A'
  const sign = delta >= 0 ? '+' : ''
  return `${sign}${delta.toFixed(1)}`
}

export function clsx(...classes: (string | boolean | undefined | null)[]): string {
  return classes.filter(Boolean).join(' ')
}

// ── ID generation ─────────────────────────────────────────────────────────────

let _counter = 0
export function uid(): string {
  return `id-${Date.now()}-${++_counter}`
}

// ── Feature display names ─────────────────────────────────────────────────────
// All temporal variants (prev, delta, roll3, baseline, from_baseline) map to
// their root display name so the UI never shows technical suffixes.

const _DISPLAY: Record<string, string> = {
  SleepHours: 'Sleep hours',               SleepHours_prev: 'Sleep hours',
  SleepHours_delta: 'Sleep hours',         SleepHours_roll3: 'Sleep hours',
  SleepHours_baseline: 'Sleep hours',      SleepHours_from_baseline: 'Sleep hours',

  BMI: 'BMI',                              BMI_prev: 'BMI',
  BMI_delta: 'BMI',                        BMI_roll3: 'BMI',
  BMI_baseline: 'BMI',                     BMI_from_baseline: 'BMI',

  Cholesterol: 'Cholesterol',              Cholesterol_prev: 'Cholesterol',
  Cholesterol_delta: 'Cholesterol',        Cholesterol_roll3: 'Cholesterol',
  Cholesterol_baseline: 'Cholesterol',     Cholesterol_from_baseline: 'Cholesterol',

  HeartRate: 'Heart rate',                 HeartRate_prev: 'Heart rate',
  HeartRate_delta: 'Heart rate',           HeartRate_roll3: 'Heart rate',
  HeartRate_baseline: 'Heart rate',        HeartRate_from_baseline: 'Heart rate',

  BiomarkerScore: 'Biomarker score',       BiomarkerScore_prev: 'Biomarker score',
  BiomarkerScore_delta: 'Biomarker score', BiomarkerScore_roll3: 'Biomarker score',
  BiomarkerScore_baseline: 'Biomarker score', BiomarkerScore_from_baseline: 'Biomarker score',

  MedicationDose: 'Medication dose',       MedicationDose_prev: 'Medication dose',
  MedicationDose_delta: 'Medication dose', MedicationDose_roll3: 'Medication dose',
  MedicationDose_baseline: 'Medication dose', MedicationDose_from_baseline: 'Medication dose',

  StressLevel: 'Stress level',             StressLevel_prev: 'Stress level',
  StressLevel_delta: 'Stress level',       StressLevel_roll3: 'Stress level',
  StressLevel_baseline: 'Stress level',    StressLevel_from_baseline: 'Stress level',

  MedicationAdherence_enc: 'Medication adherence',
  MoodScore:               'Mood score',
  StepsPerDay:             'Steps per day',
  BloodPressure_Systolic:  'Systolic BP',
  BloodPressure_Diastolic: 'Diastolic BP',
  Smoker:                  'Smoker',
  Age:                     'Age',
  days_since_first:        'Time in care',
  gap_days:                'Visit gap (days)',
  log_gap_days:            'Visit gap (log)',
  is_long_gap:             'Long visit gap',
  visit_index:             'Visit number',
}

const _SUPPRESS = new Set([
  'CognitiveScore', 'CognitiveScore_prev', 'CognitiveScore_delta',
  'CognitiveScore_roll3', 'CognitiveScore_baseline', 'CognitiveScore_from_baseline',
])

export function featureDisplayName(feature: string): string {
  return _DISPLAY[feature] ?? feature.replace(/_/g, ' ')
}

export function featureSuppressed(feature: string): boolean {
  return _SUPPRESS.has(feature)
}
