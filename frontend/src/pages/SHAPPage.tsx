import { useState, useEffect, useMemo } from 'react'
import { X, Maximize2, ArrowUpRight, ArrowDownRight, BarChart3, Info } from 'lucide-react'
import { fetchPlots, fetchFeatureImportance, plotUrl } from '../services/api'
import type { PatientData, FeatureImportanceRow } from '../types'
import Card, { CardHeader, CardTitle } from '../components/ui/Card'
import Spinner from '../components/ui/Spinner'
import { shapBarColor, featureDisplayName, featureSuppressed } from '../utils'
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, Cell } from 'recharts'

interface Props { patient: PatientData | null }

const PLOT_LABELS: Record<string, string> = {
  'shap_bar.png':              'SHAP Feature Importance',
  'shap_beeswarm.png':         'SHAP Beeswarm Plot',
  'shap_waterfall_sample.png': 'SHAP Waterfall (Sample)',
  'shap_dependence.png':       'SHAP Dependence Plot',
  'confusion_matrix.png':      'Confusion Matrix',
  'residual_analysis.png':     'Residual Analysis',
}

const PLOT_DESCRIPTIONS: Record<string, string> = {
  'shap_bar.png':              'Mean absolute SHAP value per feature — higher bars indicate greater global influence on predictions.',
  'shap_beeswarm.png':         'Each dot is one patient-visit. Colour = feature value (red=high, blue=low). Width = density.',
  'shap_waterfall_sample.png': 'Waterfall for a single prediction: how each feature pushes the score up or down from the base rate.',
  'shap_dependence.png':       "How one feature's SHAP value changes across its value range, with interaction colour-coding.",
  'confusion_matrix.png':      'Predicted vs. actual trajectory labels on the held-out test set.',
  'residual_analysis.png':     'Prediction error distribution — checks for systematic bias in the model.',
}

// ── Plot card with lightbox ───────────────────────────────────────────────────
function PlotCard({ src, label, description }: { src: string; label: string; description?: string }) {
  const [loaded, setLoaded]   = useState(false)
  const [error, setError]     = useState(false)
  const [lightbox, setLightbox] = useState(false)

  return (
    <>
      <Card hoverable padding="none" className="overflow-hidden" onClick={() => loaded && !error && setLightbox(true)}>
        {/* Card header */}
        <div className="px-4 py-3 border-b border-slate-100" onClick={e => e.stopPropagation()}>
          <div className="flex items-start justify-between gap-2">
            <div>
              <h4 className="text-sm font-semibold text-slate-800">{label}</h4>
              {description && (
                <p className="text-xs text-slate-500 mt-0.5 leading-relaxed line-clamp-2">{description}</p>
              )}
            </div>
            <button
              onClick={e => { e.stopPropagation(); if (loaded && !error) setLightbox(true) }}
              className="flex-shrink-0 p-1.5 rounded-lg text-slate-400 hover:text-indigo-600
                         hover:bg-indigo-50 transition-all duration-150"
              title="Full screen"
            >
              <Maximize2 size={13} />
            </button>
          </div>
        </div>

        {/* Image area */}
        <div className="relative bg-slate-50 min-h-[180px] flex items-center justify-center">
          {!loaded && !error && (
            <div className="absolute inset-0 flex items-center justify-center">
              <Spinner size="lg" />
            </div>
          )}
          {error ? (
            <div className="flex flex-col items-center gap-2 py-10 text-slate-400">
              <BarChart3 size={28} />
              <p className="text-xs">Plot unavailable</p>
            </div>
          ) : (
            <img
              src={src}
              alt={label}
              className="w-full object-cover object-top transition-opacity duration-300"
              style={{
                display: 'block',
                maxHeight: 240,
                opacity: loaded ? 1 : 0,
              }}
              onLoad={() => setLoaded(true)}
              onError={() => { setError(true); setLoaded(true) }}
            />
          )}
          {loaded && !error && (
            <div className="absolute inset-0 bg-gradient-to-t from-black/10 to-transparent
                            opacity-0 hover:opacity-100 transition-opacity duration-200
                            flex items-end justify-end p-2 pointer-events-none">
              <span className="bg-black/50 text-white text-[10px] rounded-lg px-2 py-1 backdrop-blur-sm">
                Click to expand
              </span>
            </div>
          )}
        </div>
      </Card>

      {/* Lightbox */}
      {lightbox && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-sm p-4"
          onClick={() => setLightbox(false)}
        >
          <div
            className="relative bg-white rounded-2xl shadow-2xl max-w-[92vw] max-h-[92vh] overflow-auto"
            onClick={e => e.stopPropagation()}
          >
            <div className="flex items-center justify-between px-5 py-4 border-b border-slate-100">
              <div>
                <h3 className="text-sm font-semibold text-slate-800">{label}</h3>
                {description && <p className="text-xs text-slate-500 mt-0.5">{description}</p>}
              </div>
              <button
                onClick={() => setLightbox(false)}
                className="p-1.5 rounded-xl text-slate-400 hover:text-slate-700 hover:bg-slate-100 transition-all"
              >
                <X size={16} />
              </button>
            </div>
            <div className="p-5">
              <img src={src} alt={label} className="block max-w-full h-auto rounded-lg" style={{ minWidth: 500 }} />
            </div>
          </div>
        </div>
      )}
    </>
  )
}

export default function SHAPPage({ patient }: Props) {
  const [plots, setPlots]           = useState<string[]>([])
  const [importance, setImportance] = useState<FeatureImportanceRow[]>([])
  const [loading, setLoading]       = useState(true)

  useEffect(() => {
    Promise.all([
      fetchPlots().then(setPlots),
      fetchFeatureImportance().then(setImportance),
    ]).finally(() => setLoading(false))
  }, [])

  const shapFactors = useMemo(() => {
    const raw  = patient?.shap_factors ?? []
    const seen = new Map<string, typeof raw[0]>()
    for (const f of raw) {
      const key = f.display_name || featureDisplayName(f.feature)
      const existing = seen.get(key)
      if (!existing || Math.abs(f.shap_value) > Math.abs(existing.shap_value))
        seen.set(key, { ...f, display_name: key })
    }
    return Array.from(seen.values())
  }, [patient?.shap_factors])

  const topImportance = useMemo(() => {
    const groups = new Map<string, number>()
    for (const row of importance) {
      if (featureSuppressed(row.feature)) continue
      const name = featureDisplayName(row.feature)
      groups.set(name, (groups.get(name) ?? 0) + row.mean_abs_shap)
    }
    return Array.from(groups.entries())
      .sort((a, b) => b[1] - a[1])
      .slice(0, 20)
      .map(([name, score]) => ({ displayName: name, score }))
  }, [importance])

  const SHAP_EXCLUDE   = new Set(['shap_waterfall_sample.png', 'shap_dependence.png'])
  const availablePlots = plots.filter(p => PLOT_LABELS[p])
  const shapPlots      = availablePlots.filter(p => p.startsWith('shap_') && !SHAP_EXCLUDE.has(p))
  const modelPlots     = availablePlots.filter(p => !p.startsWith('shap_'))

  // Interactive SHAP bar chart data
  const shapChartData = shapFactors.map(f => ({
    name:      f.display_name,
    value:     Math.abs(f.shap_value),
    direction: f.direction,
    raw:       f.shap_value,
  }))

  return (
    <div className="space-y-6 animate-fade-in">

      {/* ── Patient-specific SHAP (interactive chart) ── */}
      {patient && shapFactors.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle icon={<BarChart3 size={14} />}>
              Patient-Specific SHAP
            </CardTitle>
            <span className="text-xs text-slate-400">{patient.patient_id}</span>
          </CardHeader>

          <div className="flex items-start gap-2 p-3 rounded-xl bg-indigo-50/50 border border-indigo-100 mb-4">
            <Info size={13} className="text-indigo-500 flex-shrink-0 mt-0.5" />
            <p className="text-xs text-slate-600 leading-relaxed">
              SHAP values show how each feature pushed this prediction above or below the baseline.
              <span className="text-emerald-600 font-medium"> Green = raised</span> the score;
              <span className="text-rose-600 font-medium"> Red = lowered</span> it.
            </p>
          </div>

          {/* Horizontal bar chart */}
          <div className="h-56 mb-4">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={shapChartData} layout="vertical" margin={{ top: 0, right: 60, left: 110, bottom: 0 }}>
                <XAxis
                  type="number"
                  tick={{ fontSize: 10, fill: '#94A3B8' }}
                  axisLine={false}
                  tickLine={false}
                  tickFormatter={v => v.toFixed(2)}
                />
                <YAxis
                  type="category"
                  dataKey="name"
                  tick={{ fontSize: 11, fill: '#475569' }}
                  axisLine={false}
                  tickLine={false}
                  width={105}
                />
                <Tooltip
                  formatter={(_v, _, props) => [
                    `${props.payload.raw > 0 ? '+' : ''}${props.payload.raw.toFixed(4)}`,
                    'SHAP value',
                  ]}
                  contentStyle={{ fontSize: 11, borderRadius: 10, border: '1px solid #E2E8F0' }}
                />
                <Bar dataKey="value" radius={[0, 4, 4, 0]} maxBarSize={12}>
                  {shapChartData.map((entry, i) => (
                    <Cell key={i} fill={shapBarColor(entry.direction)} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Row-by-row SHAP bars */}
          <div className="space-y-1.5">
            {shapFactors.map((f, i) => {
              const maxAbs = Math.max(...shapFactors.map(x => Math.abs(x.shap_value)))
              const pct    = maxAbs > 0 ? (Math.abs(f.shap_value) / maxAbs) * 100 : 0
              const color  = shapBarColor(f.direction)
              const isPos  = f.direction === 'increases'
              return (
                <div key={i} className="flex items-center gap-3">
                  <div
                    className="flex-shrink-0 w-5 h-5 rounded-md flex items-center justify-center"
                    style={{ backgroundColor: isPos ? '#ECFDF5' : '#FFF1F2' }}
                  >
                    {isPos
                      ? <ArrowUpRight size={11} className="text-emerald-600" />
                      : <ArrowDownRight size={11} className="text-rose-600" />}
                  </div>
                  <div className="w-32 flex-shrink-0">
                    <span className="text-xs text-slate-700">{f.display_name}</span>
                  </div>
                  <div className="flex-1 h-3 bg-slate-100 rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-700"
                      style={{ width: `${pct}%`, backgroundColor: color, opacity: 0.8 }}
                    />
                  </div>
                  <span className="text-[11px] font-mono text-slate-500 w-14 text-right flex-shrink-0">
                    {f.shap_value > 0 ? '+' : ''}{f.shap_value.toFixed(4)}
                  </span>
                </div>
              )
            })}
          </div>
        </Card>
      )}

      {/* ── Global SHAP plots ── */}
      {loading ? (
        <div className="flex items-center gap-3 p-6 text-slate-500">
          <Spinner size="md" /> <span className="text-sm">Loading visualisations…</span>
        </div>
      ) : (
        <>
          {shapPlots.length > 0 && (
            <div className="space-y-3">
              <h3 className="text-sm font-semibold text-slate-700">Global SHAP Explanations</h3>
              <div className="grid grid-cols-2 gap-4">
                {shapPlots.map(plot => (
                  <PlotCard
                    key={plot}
                    src={plotUrl(plot)}
                    label={PLOT_LABELS[plot] || plot}
                    description={PLOT_DESCRIPTIONS[plot]}
                  />
                ))}
              </div>
            </div>
          )}

          {modelPlots.length > 0 && (
            <div className="space-y-3">
              <h3 className="text-sm font-semibold text-slate-700">Model Performance</h3>
              <div className="grid grid-cols-2 gap-4">
                {modelPlots.map(plot => (
                  <PlotCard
                    key={plot}
                    src={plotUrl(plot)}
                    label={PLOT_LABELS[plot] || plot}
                    description={PLOT_DESCRIPTIONS[plot]}
                  />
                ))}
              </div>
            </div>
          )}
        </>
      )}

      {/* ── Feature Importance Table ── */}
      {topImportance.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle icon={<BarChart3 size={14} />}>Feature Importance — Top 20</CardTitle>
            <span className="text-xs text-slate-400">Mean absolute SHAP</span>
          </CardHeader>
          <div className="overflow-x-auto rounded-xl border border-slate-100">
            <table className="w-full text-xs">
              <thead>
                <tr className="bg-slate-50 border-b border-slate-100">
                  <th className="text-left py-2.5 px-3 text-slate-500 font-semibold w-8">#</th>
                  <th className="text-left py-2.5 px-3 text-slate-500 font-semibold">Feature</th>
                  <th className="text-right py-2.5 px-3 text-slate-500 font-semibold w-24">Importance</th>
                  <th className="py-2.5 px-3 w-40"></th>
                </tr>
              </thead>
              <tbody>
                {topImportance.map((row, i) => {
                  const maxImp = topImportance[0].score
                  const pct    = (row.score / maxImp) * 100
                  return (
                    <tr
                      key={row.displayName}
                      className="border-b border-slate-50 hover:bg-slate-50/70 transition-colors"
                    >
                      <td className="py-2 px-3 text-slate-400 font-medium">{i + 1}</td>
                      <td className="py-2 px-3 text-slate-700 font-medium">{row.displayName}</td>
                      <td className="py-2 px-3 text-right font-mono text-slate-500">
                        {row.score.toFixed(4)}
                      </td>
                      <td className="py-2 px-3">
                        <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-indigo-500 rounded-full transition-all duration-500"
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  )
}
