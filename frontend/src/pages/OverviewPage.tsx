import {
  TrendingUp, TrendingDown, Minus, Target, Dna, ShieldCheck,
  AlertTriangle, CheckCircle2, HeartPulse, ArrowUpRight,
  ArrowDownRight, Clock, Users, Zap, Activity, BarChart3,
} from 'lucide-react'
import type { PatientData } from '../types'
import Card, { CardHeader, CardTitle } from '../components/ui/Card'
import Badge from '../components/ui/Badge'
import { trajectoryFill, shapBarColor, trendIcon, trendColor, fmtDelta, clsx, featureDisplayName } from '../utils'
import {
  ResponsiveContainer, PieChart, Pie, Cell, Tooltip,
  LineChart, Line, XAxis, YAxis, ReferenceLine,
} from 'recharts'

interface Props { patient: PatientData }

// ── Trajectory helpers ────────────────────────────────────────────────────────
function TrajIcon({ label, size = 14 }: { label: string; size?: number }) {
  if (label === 'Improving')     return <TrendingUp  size={size} />
  if (label === 'Deteriorating') return <TrendingDown size={size} />
  return <Minus size={size} />
}
function trajVariant(label: string): 'improving' | 'stable' | 'deteriorating' {
  if (label === 'Improving')     return 'improving'
  if (label === 'Deteriorating') return 'deteriorating'
  return 'stable'
}

// ── Score ring ────────────────────────────────────────────────────────────────
function ScoreRing({ score, label, max = 30 }: { score: number; label: string; max?: number }) {
  const pct    = Math.min((score / max) * 100, 100)
  const r      = 52
  const circ   = 2 * Math.PI * r
  const offset = circ - (pct / 100) * circ
  const color  = trajectoryFill(label)

  return (
    <div className="relative inline-flex items-center justify-center">
      <svg width="128" height="128" viewBox="0 0 128 128" className="-rotate-90">
        <circle
          cx="64" cy="64" r={r}
          strokeWidth="8"
          stroke="#F1F5F9"
          fill="none"
        />
        <circle
          cx="64" cy="64" r={r}
          strokeWidth="8"
          stroke={color}
          fill="none"
          strokeLinecap="round"
          strokeDasharray={circ}
          strokeDashoffset={offset}
          style={{ transition: 'stroke-dashoffset 1s ease-out' }}
        />
      </svg>
      <div className="absolute flex flex-col items-center">
        <span className="text-3xl font-bold text-slate-900 leading-none">{score.toFixed(1)}</span>
        <span className="text-xs text-slate-500 mt-1">/ {max}</span>
      </div>
    </div>
  )
}

// ── Metric card ───────────────────────────────────────────────────────────────
function MetricCard({
  label, value, sub, trend,
}: {
  label: string; value: string; sub?: string; trend?: 'up' | 'down' | 'neutral'
}) {
  return (
    <div className="rounded-xl p-3 bg-slate-50 border border-slate-100">
      <p className="text-[11px] font-medium text-slate-500 mb-1.5">{label}</p>
      <div className="flex items-baseline gap-1.5">
        <p className="text-base font-bold text-slate-900">{value}</p>
        {trend === 'up' && <ArrowUpRight size={12} className="text-emerald-500" />}
        {trend === 'down' && <ArrowDownRight size={12} className="text-rose-500" />}
      </div>
      {sub && <p className="text-[10px] text-slate-400 mt-0.5">{sub}</p>}
    </div>
  )
}

// ── SHAP bar row ──────────────────────────────────────────────────────────────
function ShapBar({ label, value, direction }: { label: string; value: number; direction: string }) {
  const maxAbs = 0.5
  const pct    = Math.min(Math.abs(value) / maxAbs * 100, 100)
  const color  = shapBarColor(direction)
  const isPos  = direction === 'increases'

  return (
    <div className="flex items-center gap-3 py-1.5">
      <div
        className="flex-shrink-0 w-5 h-5 rounded-md flex items-center justify-center"
        style={{ backgroundColor: isPos ? '#ECFDF5' : '#FFF1F2' }}
      >
        {isPos
          ? <ArrowUpRight size={11} className="text-emerald-600" />
          : <ArrowDownRight size={11} className="text-rose-600" />}
      </div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between mb-1">
          <span className="text-xs text-slate-700 truncate">{label}</span>
          <span className="text-xs font-mono text-slate-500 ml-2 flex-shrink-0">{Math.abs(value).toFixed(3)}</span>
        </div>
        <div className="h-1.5 rounded-full bg-slate-100 overflow-hidden">
          <div
            className="h-full rounded-full transition-all duration-700"
            style={{ width: `${pct}%`, backgroundColor: color }}
          />
        </div>
      </div>
    </div>
  )
}

// ── Trend row ─────────────────────────────────────────────────────────────────
function TrendRow({ feature, dir, mag }: { feature: string; dir: string; mag: string }) {
  const icon  = trendIcon(dir)
  const color = trendColor(dir)
  return (
    <div className="flex items-center justify-between py-2 border-b border-slate-50 last:border-0">
      <span className="text-xs text-slate-600 capitalize">{featureDisplayName(feature)}</span>
      <span className={clsx('text-xs font-medium flex items-center gap-1', color)}>
        {icon} {mag} {dir}
      </span>
    </div>
  )
}

export default function OverviewPage({ patient }: Props) {
  const cp = patient.clinical_profile
  const cr = patient.clinical_reasoning
  const r  = patient.reasoning
  const tr = patient.feature_trends

  const score  = patient.predicted_score
  const label  = patient.predicted_label
  const change = patient.predicted_change

  const trajectoryData = [
    { name: 'Improving',     value: patient.improving_pct,     fill: '#10B981' },
    { name: 'Stable',        value: patient.stable_pct,        fill: '#F59E0B' },
    { name: 'Deteriorating', value: patient.deteriorating_pct, fill: '#F43F5E' },
  ]

  const trendEntries = Object.entries(tr).filter(([, v]) => v.direction !== 'stable')

  const confidencePct = cr.confidence_level === 'High' ? 90 : cr.confidence_level === 'Moderate' ? 55 : 22

  return (
    <div className="space-y-5 animate-fade-in">

      {/* ── HERO ROW: Prediction + Phenotype + Confidence ── */}
      <div className="grid grid-cols-3 gap-4">

        {/* Prediction hero card */}
        <Card variant="default" className="col-span-1 relative overflow-hidden">
          {/* Subtle gradient stripe */}
          <div
            className="absolute top-0 left-0 right-0 h-1 rounded-t-2xl"
            style={{ background: `linear-gradient(90deg, ${trajectoryFill(label)}, ${trajectoryFill(label)}88)` }}
          />
          <CardHeader>
            <CardTitle icon={<Target size={14} />}>Prediction</CardTitle>
            <Badge variant={trajVariant(label)} dot>
              <TrajIcon label={label} size={10} /> {label}
            </Badge>
          </CardHeader>

          <div className="flex items-center gap-6">
            {score != null ? (
              <ScoreRing score={score} label={label} />
            ) : (
              <div className="text-4xl font-bold text-slate-300">N/A</div>
            )}
            <div className="flex-1 space-y-2">
              {change != null && (
                <div className={clsx(
                  'inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-sm font-bold',
                  change >= 0 ? 'bg-emerald-50 text-emerald-700' : 'bg-rose-50 text-rose-700',
                )}>
                  {change >= 0 ? <ArrowUpRight size={14} /> : <ArrowDownRight size={14} />}
                  {fmtDelta(change)} pts
                </div>
              )}
              <p className="text-xs text-slate-500 leading-relaxed">
                Predicted cognitive score from XGBoost model
              </p>
              <p className="text-[11px] text-slate-400">Range: 0–30</p>
            </div>
          </div>
        </Card>

        {/* Phenotype card */}
        <Card className="col-span-1">
          <CardHeader>
            <CardTitle icon={<Dna size={14} />}>Phenotype</CardTitle>
            <Badge variant="indigo">{patient.n_patients} pts</Badge>
          </CardHeader>
          <p className="text-xs font-semibold text-slate-800 mb-1 leading-snug">{patient.phenotype}</p>
          <p className="text-xs text-slate-500 mb-3 leading-relaxed line-clamp-2">{patient.phenotype_summary}</p>

          {/* Outcome donut */}
          <div className="flex items-center gap-3">
            <div className="h-20 w-20 flex-shrink-0">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={trajectoryData}
                    cx="50%" cy="50%"
                    innerRadius={22} outerRadius={36}
                    dataKey="value"
                    startAngle={90} endAngle={-270}
                  >
                    {trajectoryData.map((d, i) => (
                      <Cell key={i} fill={d.fill} strokeWidth={0} />
                    ))}
                  </Pie>
                  <Tooltip
                    formatter={(v) => [`${Number(v).toFixed(1)}%`, '']}
                    contentStyle={{ fontSize: 11, borderRadius: 8, border: '1px solid #E2E8F0' }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>
            <div className="space-y-1 flex-1">
              {trajectoryData.map(d => (
                <div key={d.name} className="flex items-center justify-between">
                  <div className="flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full flex-shrink-0" style={{ backgroundColor: d.fill }} />
                    <span className="text-[11px] text-slate-600">{d.name}</span>
                  </div>
                  <span className="text-[11px] font-semibold text-slate-800">{d.value.toFixed(0)}%</span>
                </div>
              ))}
            </div>
          </div>
        </Card>

        {/* Confidence card */}
        <Card className="col-span-1">
          <CardHeader>
            <CardTitle icon={<ShieldCheck size={14} />}>Confidence</CardTitle>
            <Badge variant={cr.confidence_level === 'High' ? 'high' : cr.confidence_level === 'Moderate' ? 'moderate' : 'low'}>
              {cr.confidence_level}
            </Badge>
          </CardHeader>

          {/* Gauge */}
          <div className="mb-3">
            <div className="flex justify-between text-[10px] text-slate-400 mb-1.5">
              <span>Low</span><span>Moderate</span><span>High</span>
            </div>
            <div className="h-2 rounded-full bg-slate-100 overflow-hidden">
              <div
                className="h-full rounded-full transition-all duration-1000"
                style={{
                  width: `${confidencePct}%`,
                  background: cr.confidence_level === 'High'
                    ? 'linear-gradient(90deg, #10B981, #059669)'
                    : cr.confidence_level === 'Moderate'
                      ? 'linear-gradient(90deg, #F59E0B, #D97706)'
                      : 'linear-gradient(90deg, #F43F5E, #E11D48)',
                }}
              />
            </div>
          </div>

          <p className="text-xs text-slate-600 leading-relaxed mb-3">{cr.confidence_statement}</p>

          <div className="flex items-center gap-2 pt-3 border-t border-slate-100">
            <Users size={12} className="text-slate-400 flex-shrink-0" />
            <p className="text-xs text-slate-500">
              <span className="font-semibold text-slate-800">{r.n_peers}</span> similar patients compared
            </p>
          </div>
        </Card>
      </div>

      {/* ── COGNITIVE SCORE HISTORY ── */}
      {patient.cognitive_score_history && patient.cognitive_score_history.length > 1 && (
        <Card>
          <CardHeader>
            <CardTitle icon={<TrendingUp size={14} />}>Cognitive Score History</CardTitle>
            <span className="text-xs text-slate-400">
              {patient.cognitive_score_history.length} visits recorded
            </span>
          </CardHeader>
          <div className="h-32">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart
                data={patient.cognitive_score_history}
                margin={{ top: 4, right: 12, left: -24, bottom: 0 }}
              >
                <XAxis
                  dataKey="date"
                  tick={{ fontSize: 10, fill: '#94A3B8' }}
                  tickFormatter={(d: string) => {
                    const date = new Date(d)
                    return date.toLocaleDateString('en-GB', { month: 'short', year: '2-digit' })
                  }}
                  interval="preserveStartEnd"
                  axisLine={false}
                  tickLine={false}
                />
                <YAxis
                  domain={[0, 30]}
                  tick={{ fontSize: 10, fill: '#94A3B8' }}
                  tickCount={4}
                  axisLine={false}
                  tickLine={false}
                />
                <Tooltip
                  formatter={(v) => [typeof v === 'number' ? v.toFixed(1) : v, 'Cognitive Score']}
                  labelFormatter={(d) => `Visit: ${String(d).slice(0, 10)}`}
                  contentStyle={{ fontSize: 11, borderRadius: 8, border: '1px solid #E2E8F0' }}
                />
                <ReferenceLine y={score ?? undefined} stroke={trajectoryFill(label)} strokeDasharray="4 2" strokeOpacity={0.4} />
                <Line
                  type="monotone"
                  dataKey="score"
                  stroke={trajectoryFill(label)}
                  strokeWidth={2}
                  dot={{ r: 3, fill: trajectoryFill(label), strokeWidth: 0 }}
                  activeDot={{ r: 5 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>
      )}

      {/* ── CLINICAL + SHAP ── */}
      <div className="grid grid-cols-2 gap-4">

        {/* Clinical Snapshot */}
        <Card>
          <CardHeader>
            <CardTitle icon={<HeartPulse size={14} />}>Clinical Snapshot</CardTitle>
            <span className="text-xs text-slate-400 flex items-center gap-1">
              <Clock size={11} /> Most recent visit
            </span>
          </CardHeader>

          <div className="grid grid-cols-3 gap-2 mb-3">
            <MetricCard label="Age"         value={cp.age}                              sub={cp.gender} />
            <MetricCard label="BMI"         value={cp.bmi}                              sub="kg/m²" />
            <MetricCard label="Cholesterol" value={`${cp.cholesterol}`}                 sub="mg/dL" />
            <MetricCard label="Heart Rate"  value={`${cp.heart_rate}`}                  sub="bpm" />
            <MetricCard label="Blood Press" value={`${cp.bp_systolic}/${cp.bp_diastolic}`} sub="mmHg" />
            <MetricCard label="Sleep"       value={`${cp.sleep_hours}h`}                sub="per night" />
          </div>

          <div className="space-y-1 pt-3 border-t border-slate-100">
            {[
              { label: 'Cognitive score', value: `${cp.cognitive_score}  (Δ ${cp.cognitive_delta})` },
              { label: 'Stress level',    value: `${cp.stress_level}/9` },
              { label: 'Steps/day',       value: cp.steps_per_day },
              { label: 'Adherence',       value: cp.medication_adherence },
              { label: 'Disease',         value: cp.disease },
              { label: 'Stage',           value: cp.stage },
              { label: 'Lifestyle',       value: cp.lifestyle },
            ].map(({ label, value }) => (
              <div key={label} className="flex items-center justify-between py-0.5">
                <span className="text-xs text-slate-400">{label}</span>
                <span className="text-xs font-medium text-slate-700">{value || 'N/A'}</span>
              </div>
            ))}
          </div>
        </Card>

        {/* SHAP Factors */}
        <Card>
          <CardHeader>
            <CardTitle icon={<BarChart3 size={14} />}>Top Model Drivers</CardTitle>
            <span className="text-xs text-slate-400">SHAP values</span>
          </CardHeader>

          <div className="space-y-0.5 mb-3">
            {patient.shap_factors.map((f, i) => (
              <ShapBar key={i} label={f.display_name} value={f.shap_value} direction={f.direction} />
            ))}
          </div>

          <div className="flex gap-4 pt-3 border-t border-slate-100">
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-sm bg-emerald-500" />
              <span className="text-[11px] text-slate-500">Improves score</span>
            </div>
            <div className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-sm bg-rose-500" />
              <span className="text-[11px] text-slate-500">Reduces score</span>
            </div>
          </div>
        </Card>
      </div>

      {/* ── PRIORITIES + CONCERNS + TRENDS ── */}
      <div className="grid grid-cols-3 gap-4">

        {/* Actionable Priorities */}
        <Card>
          <CardHeader>
            <CardTitle icon={<Zap size={14} />}>Actionable Priorities</CardTitle>
          </CardHeader>
          <ol className="space-y-2.5">
            {cr.actionable_priorities.map((p, i) => (
              <li key={i} className="flex gap-3">
                <span className="flex-shrink-0 w-5 h-5 rounded-lg bg-indigo-50 text-indigo-600 text-[10px]
                                 font-bold flex items-center justify-center mt-0.5">
                  {i + 1}
                </span>
                <p className="text-xs text-slate-700 leading-relaxed">{p}</p>
              </li>
            ))}
          </ol>
        </Card>

        {/* Major Concerns + Protective Factors */}
        <Card>
          <CardHeader>
            <CardTitle icon={<AlertTriangle size={14} />}>Clinical Concerns</CardTitle>
          </CardHeader>
          <ul className="space-y-2 mb-3">
            {cr.major_concerns.slice(0, 4).map((c, i) => (
              <li key={i} className="flex gap-2.5">
                <span className="flex-shrink-0 mt-1 w-1.5 h-1.5 rounded-full bg-rose-400" />
                <p className="text-xs text-slate-700 leading-relaxed">{c}</p>
              </li>
            ))}
          </ul>
          {cr.protective_factors.length > 0 && (
            <div className="pt-3 border-t border-slate-100">
              <p className="text-[11px] font-semibold text-emerald-600 mb-2 flex items-center gap-1">
                <CheckCircle2 size={11} /> Protective Factors
              </p>
              <ul className="space-y-1.5">
                {cr.protective_factors.slice(0, 2).map((p, i) => (
                  <li key={i} className="flex gap-2.5">
                    <span className="flex-shrink-0 mt-1 w-1.5 h-1.5 rounded-full bg-emerald-400" />
                    <p className="text-xs text-slate-600 leading-relaxed">{p}</p>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </Card>

        {/* Feature Trends */}
        <Card>
          <CardHeader>
            <CardTitle icon={<Activity size={14} />}>Feature Trends</CardTitle>
            <span className="text-xs text-slate-400">Recent visits</span>
          </CardHeader>
          {trendEntries.length > 0 ? (
            <div>
              {trendEntries.map(([feat, t]) => (
                <TrendRow key={feat} feature={feat} dir={t.direction} mag={t.magnitude} />
              ))}
            </div>
          ) : (
            <p className="text-xs text-slate-400 italic">All tracked features are stable.</p>
          )}
          <div className="mt-3 pt-3 border-t border-slate-100">
            <p className="text-xs text-slate-500 italic leading-relaxed">{r.overall_summary}</p>
          </div>
        </Card>
      </div>

      {/* ── Peer Context ── */}
      {cr.peer_summary && (
        <Card>
          <CardHeader>
            <CardTitle icon={<Users size={14} />}>Peer Context</CardTitle>
          </CardHeader>
          <p className="text-sm text-slate-600 leading-relaxed mb-4">{cr.peer_summary}</p>
          {cr.recommended_follow_up_questions.length > 0 && (
            <div>
              <p className="text-[11px] font-semibold text-slate-400 uppercase tracking-wide mb-2">
                Suggested follow-up questions
              </p>
              <div className="flex flex-wrap gap-2">
                {cr.recommended_follow_up_questions.map((q, i) => (
                  <span
                    key={i}
                    className="inline-flex items-center gap-1 rounded-lg px-3 py-1.5 text-xs
                               bg-indigo-50 text-indigo-700 border border-indigo-100"
                  >
                    💡 {q}
                  </span>
                ))}
              </div>
            </div>
          )}
        </Card>
      )}
    </div>
  )
}

