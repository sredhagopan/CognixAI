import { useState, useEffect, useCallback } from 'react'
import {
  RotateCcw, Play, TrendingUp, TrendingDown, Minus,
  AlertTriangle, ArrowRight, Sparkles, FlaskConical,
  ArrowUpRight, ArrowDownRight, Info, ChevronDown, ChevronUp,
} from 'lucide-react'
import { runSimulation, fetchSimulationMeta } from '../services/api'
import type { PatientData, SimulationResult, SimulationMeta } from '../types'
import { clsx, trajectoryFill, fmtDelta, featureDisplayName } from '../utils'
import Card, { CardHeader, CardTitle } from '../components/ui/Card'
import Badge from '../components/ui/Badge'
import Spinner from '../components/ui/Spinner'
import Button from '../components/ui/Button'

interface Props { patient: PatientData }

const FEATURE_LABELS: Record<string, string> = {
  BMI:                     'BMI',
  Cholesterol:             'Cholesterol',
  HeartRate:               'Heart Rate',
  SleepHours:              'Sleep Hours',
  StepsPerDay:             'Steps per Day',
  StressLevel:             'Stress Level',
  MedicationDose:          'Medication Dose',
  MedicationAdherence_enc: 'Medication Adherence',
  BiomarkerScore:          'Biomarker Score',
  MoodScore:               'Mood Score',
  BloodPressure_Systolic:  'Systolic BP',
  BloodPressure_Diastolic: 'Diastolic BP',
  Smoker:                  'Smoker',
}

const FEATURE_UNITS: Record<string, string> = {
  Cholesterol:             'mg/dL',
  HeartRate:               'bpm',
  SleepHours:              'hrs',
  StepsPerDay:             'steps/day',
  BloodPressure_Systolic:  'mmHg',
  BloodPressure_Diastolic: 'mmHg',
  BMI:                     'kg/m²',
  StressLevel:             '/9',
  MoodScore:               '/9',
}

const ADHERENCE_LABELS: Record<number, string> = { 0: 'Low', 1: 'Medium', 2: 'High' }

function getPatientCurrentValue(feature: string, patient: PatientData): number | null {
  const cp = patient.clinical_profile
  const map: Record<string, string | null> = {
    BMI:                     cp.bmi,
    Cholesterol:             cp.cholesterol,
    HeartRate:               cp.heart_rate,
    SleepHours:              cp.sleep_hours,
    StepsPerDay:             cp.steps_per_day,
    StressLevel:             cp.stress_level,
    MedicationDose:          cp.medication_dose,
    BiomarkerScore:          cp.biomarker_score,
    MoodScore:               cp.mood_score,
    BloodPressure_Systolic:  cp.bp_systolic,
    BloodPressure_Diastolic: cp.bp_diastolic,
  }
  const v = map[feature]
  if (v == null || v === 'N/A') return null
  const n = parseFloat(v)
  return isNaN(n) ? null : n
}

function deduplicateShifts(shifts: Array<{ feature: string; display_name?: string; shift: number }>) {
  const seen = new Map<string, typeof shifts[0]>()
  for (const s of shifts) {
    const key = s.display_name || featureDisplayName(s.feature)
    const existing = seen.get(key)
    if (!existing || Math.abs(s.shift) > Math.abs(existing.shift)) seen.set(key, s)
  }
  return Array.from(seen.values())
}

// ── Premium Slider ────────────────────────────────────────────────────────────
interface SliderRowProps {
  feature:    string
  targets:    number[]
  value:      number
  original:   number | null
  onChange:   (v: number) => void
  onReset:    () => void
  isAdherence: boolean
}

function SliderRow({ feature, targets, value, original, onChange, onReset, isAdherence }: SliderRowProps) {
  const label    = FEATURE_LABELS[feature] || feature
  const unit     = FEATURE_UNITS[feature] || ''
  const min      = targets[0]
  const max      = targets[targets.length - 1]
  const step     = isAdherence ? 1 : targets.length > 1
    ? Math.min(...targets.slice(1).map((v, i) => v - targets[i]))
    : 1
  const hasChanged = original !== null && value !== original
  const displayVal = isAdherence ? (ADHERENCE_LABELS[value] || String(value)) : value
  const displayOrig = original !== null
    ? (isAdherence ? ADHERENCE_LABELS[original] : original)
    : null
  const pct = max > min ? ((value - min) / (max - min)) * 100 : 0

  return (
    <div className={clsx(
      'rounded-xl p-4 border transition-all duration-200',
      hasChanged
        ? 'bg-indigo-50/50 border-indigo-200'
        : 'bg-white border-slate-200 card-shadow',
    )}>
      <div className="flex items-start justify-between mb-3">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-sm font-medium text-slate-800">{label}</span>
            {hasChanged && (
              <Badge variant="indigo" className="text-[10px] px-1.5 py-0.5">modified</Badge>
            )}
          </div>
          {unit && <p className="text-[10px] text-slate-400 mt-0.5">{unit}</p>}
        </div>
        <div className="flex items-center gap-2 text-right">
          <div>
            <span className={clsx(
              'text-base font-bold block',
              hasChanged ? 'text-indigo-600' : 'text-slate-800',
            )}>
              {displayVal}
            </span>
            {hasChanged && displayOrig !== null && (
              <span className="text-[10px] text-slate-400">was {displayOrig}</span>
            )}
          </div>
          {hasChanged && (
            <button
              onClick={onReset}
              className="text-slate-400 hover:text-slate-600 p-1 rounded-lg hover:bg-white transition-all"
              title="Reset to original"
            >
              <RotateCcw size={12} />
            </button>
          )}
        </div>
      </div>

      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full"
        style={{
          background: `linear-gradient(to right, ${hasChanged ? '#6366F1' : '#CBD5E1'} 0%, ${hasChanged ? '#6366F1' : '#CBD5E1'} ${pct}%, #E2E8F0 ${pct}%, #E2E8F0 100%)`,
        }}
      />

      <div className="flex justify-between text-[10px] text-slate-400 mt-1">
        <span>{isAdherence ? ADHERENCE_LABELS[min] : min}</span>
        <span>{isAdherence ? ADHERENCE_LABELS[max] : max}</span>
      </div>
    </div>
  )
}

// ── Score transition display ──────────────────────────────────────────────────
function ScoreTransition({ before, after, delta, beforeLabel, afterLabel, labelChanged }: {
  before: number
  after:  number
  delta:  number
  beforeLabel: string
  afterLabel:  string
  labelChanged: boolean
}) {
  const improved = delta > 0
  const minimal  = Math.abs(delta) < 0.5

  function TrajBadge({ label }: { label: string }) {
    const color = trajectoryFill(label)
    const Icon = label === 'Improving' ? TrendingUp : label === 'Deteriorating' ? TrendingDown : Minus
    return (
      <span className="inline-flex items-center gap-1 text-xs font-medium px-2.5 py-1 rounded-full"
            style={{ backgroundColor: `${color}18`, color }}>
        <Icon size={11} /> {label}
      </span>
    )
  }

  return (
    <div className="space-y-3">
      {/* Before → After scores */}
      <div className="grid grid-cols-3 gap-3">
        <div className="rounded-xl p-4 text-center bg-slate-50 border border-slate-200">
          <p className="text-[11px] text-slate-400 mb-2 font-medium">Before</p>
          <p className="text-3xl font-bold text-slate-800">{before.toFixed(1)}</p>
          <div className="mt-2 flex justify-center">
            <TrajBadge label={beforeLabel} />
          </div>
        </div>

        <div className={clsx(
          'rounded-xl p-4 text-center border flex flex-col items-center justify-center',
          minimal  ? 'bg-slate-50 border-slate-200'      :
          improved ? 'bg-emerald-50 border-emerald-200'  : 'bg-rose-50 border-rose-200',
        )}>
          {!minimal && (
            improved
              ? <ArrowUpRight size={20} className="text-emerald-600 mb-1" />
              : <ArrowDownRight size={20} className="text-rose-600 mb-1" />
          )}
          <p className={clsx(
            'text-3xl font-bold',
            minimal  ? 'text-slate-600'   :
            improved ? 'text-emerald-700' : 'text-rose-700',
          )}>
            {fmtDelta(delta)}
          </p>
          <p className="text-[10px] text-slate-400 mt-0.5">pts change</p>
          {labelChanged && (
            <div className="mt-2 flex items-center gap-1 text-[10px] font-medium">
              <span style={{ color: trajectoryFill(beforeLabel) }}>{beforeLabel}</span>
              <ArrowRight size={10} className="text-slate-400" />
              <span style={{ color: trajectoryFill(afterLabel) }}>{afterLabel}</span>
            </div>
          )}
        </div>

        <div className={clsx(
          'rounded-xl p-4 text-center border',
          labelChanged
            ? (improved ? 'bg-emerald-50 border-emerald-200' : 'bg-rose-50 border-rose-200')
            : 'bg-slate-50 border-slate-200',
        )}>
          <p className="text-[11px] text-slate-400 mb-2 font-medium">After</p>
          <p className="text-3xl font-bold text-slate-800">{after.toFixed(1)}</p>
          <div className="mt-2 flex justify-center">
            <TrajBadge label={afterLabel} />
          </div>
        </div>
      </div>

      {minimal && (
        <div className="rounded-xl p-3 bg-amber-50 border border-amber-200 flex items-start gap-2">
          <Info size={13} className="text-amber-600 flex-shrink-0 mt-0.5" />
          <p className="text-xs text-amber-700">
            <span className="font-semibold">Minimal effect:</span> These features are not dominant model drivers.
            Score changed by less than 0.5 points.
          </p>
        </div>
      )}
    </div>
  )
}

export default function SimulationPage({ patient }: Props) {
  const [meta, setMeta]           = useState<SimulationMeta | null>(null)
  const [values, setValues]       = useState<Record<string, number>>({})
  const [originals, setOriginals] = useState<Record<string, number | null>>({})
  const [result, setResult]       = useState<SimulationResult | null>(null)
  const [loading, setLoading]     = useState(false)
  const [error, setError]         = useState<string | null>(null)
  const [showShaps, setShowShaps] = useState(false)

  useEffect(() => {
    fetchSimulationMeta().then(setMeta).catch(() => {})
  }, [])

  useEffect(() => {
    if (!meta) return
    const init:  Record<string, number>         = {}
    const origs: Record<string, number | null>  = {}
    Object.keys(meta.realistic_targets).forEach((feat) => {
      const current = getPatientCurrentValue(feat, patient)
      const targets = meta.realistic_targets[feat]
      const defaultVal = current != null
        ? targets.reduce((a, b) => Math.abs(b - current) < Math.abs(a - current) ? b : a)
        : targets[Math.floor(targets.length / 2)]
      init[feat]  = defaultVal
      origs[feat] = current
    })
    setValues(init)
    setOriginals(origs)
    setResult(null)
  }, [meta, patient.patient_id])

  const handleRun = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const changes: Record<string, number> = {}
      Object.entries(values).forEach(([feat, val]) => {
        if (originals[feat] !== val) changes[feat] = val
      })
      if (Object.keys(changes).length === 0) {
        setError('No changes detected. Adjust at least one slider before running.')
        setLoading(false)
        return
      }
      const res = await runSimulation(patient.patient_id, 'specific', changes)
      setResult(res)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Simulation failed')
    } finally {
      setLoading(false)
    }
  }, [values, originals, patient.patient_id])

  const handleReset = useCallback(() => {
    if (!meta) return
    const reset: Record<string, number> = {}
    Object.keys(meta.realistic_targets).forEach((feat) => {
      const current = originals[feat]
      const targets = meta.realistic_targets[feat]
      reset[feat] = current != null
        ? targets.reduce((a, b) => Math.abs(b - current) < Math.abs(a - current) ? b : a)
        : targets[Math.floor(targets.length / 2)]
    })
    setValues(reset)
    setResult(null)
    setError(null)
  }, [meta, originals])

  const changedCount = Object.entries(values).filter(
    ([f, v]) => originals[f] !== null && originals[f] !== v,
  ).length

  return (
    <div className="space-y-5 animate-fade-in">

      {/* ── Header card ── */}
      <Card>
        <div className="flex items-center justify-between flex-wrap gap-4">
          <div>
            <div className="flex items-center gap-2 mb-1">
              <FlaskConical size={18} className="text-indigo-500" />
              <h2 className="text-base font-semibold text-slate-800">What-If Simulation</h2>
              {changedCount > 0 && (
                <Badge variant="indigo">{changedCount} modified</Badge>
              )}
            </div>
            <p className="text-xs text-slate-500">
              Adjust feature controls to model the impact of clinical interventions on the predicted score.
            </p>
          </div>
          <div className="flex gap-2">
            <Button variant="outline" size="sm" icon={<RotateCcw size={13} />} onClick={handleReset}>
              Reset
            </Button>
            <Button
              variant="primary"
              size="sm"
              icon={loading ? <Spinner size="xs" className="text-white" /> : <Play size={13} />}
              onClick={handleRun}
              disabled={loading}
              loading={false}
            >
              {loading ? 'Running…' : 'Run Simulation'}
            </Button>
          </div>
        </div>
      </Card>

      {/* ── Two-column layout ── */}
      <div className="grid grid-cols-2 gap-5">

        {/* Left: Feature controls */}
        <div className="space-y-3">
          <h3 className="text-sm font-semibold text-slate-700 flex items-center gap-2">
            <Sparkles size={14} className="text-indigo-500" />
            Feature Controls
          </h3>
          {meta ? (
            Object.entries(meta.realistic_targets).map(([feat, targets]) => (
              <SliderRow
                key={feat}
                feature={feat}
                targets={targets}
                value={values[feat] ?? targets[0]}
                original={originals[feat] ?? null}
                isAdherence={feat === 'MedicationAdherence_enc'}
                onChange={(v) => setValues(prev => ({ ...prev, [feat]: v }))}
                onReset={() => {
                  const current = originals[feat]
                  if (current != null) {
                    const targets2 = meta.realistic_targets[feat]
                    const snapped = targets2.reduce((a, b) =>
                      Math.abs(b - current) < Math.abs(a - current) ? b : a,
                    )
                    setValues(prev => ({ ...prev, [feat]: snapped }))
                  }
                }}
              />
            ))
          ) : (
            <div className="flex items-center gap-3 p-6 rounded-xl bg-white border border-slate-200">
              <Spinner size="md" />
              <span className="text-sm text-slate-500">Loading simulation controls…</span>
            </div>
          )}
        </div>

        {/* Right: Result panel */}
        <div className="space-y-4">
          <h3 className="text-sm font-semibold text-slate-700 flex items-center gap-2">
            <TrendingUp size={14} className="text-indigo-500" />
            Simulation Result
          </h3>

          {error && (
            <div className="rounded-xl p-3 bg-rose-50 border border-rose-200 flex items-center gap-2">
              <AlertTriangle size={14} className="text-rose-500 flex-shrink-0" />
              <p className="text-xs text-rose-700">{error}</p>
            </div>
          )}

          {result ? (
            <div className="space-y-4 animate-fade-in">
              <ScoreTransition
                before={result.before_score}
                after={result.after_score}
                delta={result.delta}
                beforeLabel={result.before_label}
                afterLabel={result.after_label}
                labelChanged={result.label_changed}
              />

              {/* Individual impacts */}
              {result.individual_impacts && result.individual_impacts.length > 0 && (
                <Card padding="sm">
                  <CardHeader>
                    <CardTitle>Individual Contributions</CardTitle>
                  </CardHeader>
                  <div className="space-y-2.5">
                    {result.individual_impacts.map((imp, i) => {
                      const maxDelta = Math.max(...result.individual_impacts!.map(x => x.delta))
                      const pct = maxDelta > 0 ? (imp.delta / maxDelta) * 100 : 0
                      return (
                        <div key={i} className="space-y-1">
                          <div className="flex items-center justify-between text-xs">
                            <span className="text-slate-600">
                              {featureDisplayName(imp.feature)} → {imp.value}
                            </span>
                            <span className="font-semibold text-emerald-600">
                              +{imp.delta.toFixed(2)} pts
                            </span>
                          </div>
                          <div className="h-1.5 rounded-full bg-slate-100 overflow-hidden">
                            <div
                              className="h-full rounded-full bg-emerald-500 transition-all duration-700"
                              style={{ width: `${pct}%` }}
                            />
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </Card>
              )}

              {/* SHAP shifts */}
              {result.shap_shifts && result.shap_shifts.length > 0 && (
                <Card padding="sm">
                  <button
                    className="w-full flex items-center justify-between text-left"
                    onClick={() => setShowShaps(v => !v)}
                  >
                    <span className="text-sm font-semibold text-slate-700">
                      Feature Contribution Shifts
                    </span>
                    {showShaps ? <ChevronUp size={14} className="text-slate-400" /> : <ChevronDown size={14} className="text-slate-400" />}
                  </button>
                  {showShaps && (
                    <div className="mt-3 space-y-1.5">
                      {deduplicateShifts(result.shap_shifts).map((s, i) => (
                        <div key={i} className="flex items-center justify-between text-xs py-1 border-b border-slate-50 last:border-0">
                          <span className="text-slate-600">{s.display_name || featureDisplayName(s.feature)}</span>
                          <span className={clsx(
                            'font-medium flex items-center gap-1',
                            s.shift > 0 ? 'text-emerald-600' : 'text-rose-600',
                          )}>
                            {s.shift > 0
                              ? <><ArrowUpRight size={11} /> increased</>
                              : <><ArrowDownRight size={11} /> decreased</>}
                          </span>
                        </div>
                      ))}
                    </div>
                  )}
                </Card>
              )}

              {/* Cascade warnings */}
              {meta && Object.keys(result.changes_applied).map((feat) => {
                const cascades = meta.feature_cascades[feat]
                if (!cascades || cascades.length === 0) return null
                return (
                  <div key={feat} className="rounded-xl p-3 bg-amber-50 border border-amber-200 flex items-start gap-2">
                    <AlertTriangle size={13} className="text-amber-600 flex-shrink-0 mt-0.5" />
                    <p className="text-xs text-amber-700">
                      <span className="font-semibold">{featureDisplayName(feat)}</span> is typically
                      linked to{' '}
                      <span className="font-medium">{cascades.map(c => featureDisplayName(c)).join(', ')}</span>{' '}
                      in real-world clinical scenarios.
                    </p>
                  </div>
                )
              })}
            </div>
          ) : (
            <div className="rounded-2xl border border-slate-200 bg-white p-10 text-center card-shadow">
              {loading ? (
                <div className="flex flex-col items-center gap-3">
                  <Spinner size="lg" />
                  <p className="text-sm text-slate-600 font-medium">Running simulation…</p>
                  <p className="text-xs text-slate-400">This may take a few seconds</p>
                </div>
              ) : (
                <>
                  <div className="w-14 h-14 rounded-2xl bg-indigo-50 flex items-center justify-center mx-auto mb-4">
                    <FlaskConical size={24} className="text-indigo-400" />
                  </div>
                  <p className="text-sm font-semibold text-slate-700 mb-1">No simulation yet</p>
                  <p className="text-xs text-slate-400">
                    Adjust the sliders on the left and click Run Simulation
                  </p>
                </>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
