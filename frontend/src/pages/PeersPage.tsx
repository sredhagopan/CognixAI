import { Users, TrendingUp, TrendingDown, Minus, AlertTriangle, ArrowUpRight, ArrowDownRight } from 'lucide-react'
import type { PatientData } from '../types'
import Card, { CardHeader, CardTitle } from '../components/ui/Card'
import Badge from '../components/ui/Badge'
import { clsx } from '../utils'
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis,
  ResponsiveContainer, Tooltip,
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Legend,
} from 'recharts'

interface Props { patient: PatientData }

// ── Percentile bar ────────────────────────────────────────────────────────────
function PercentileBar({ pct }: { pct: number }) {
  const color = pct >= 70 ? '#10B981' : pct <= 30 ? '#F43F5E' : '#F59E0B'
  return (
    <div className="mt-2">
      <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{ width: `${pct}%`, backgroundColor: color }}
        />
      </div>
    </div>
  )
}

// ── Peer stat card ────────────────────────────────────────────────────────────
function StatCard({ label, patientVal, peerMean, percentile, interpretation }: {
  label: string
  patientVal: string | number
  peerMean?: number
  percentile?: number
  interpretation?: string
}) {
  const pct          = percentile ?? 50
  const badgeVariant = pct >= 70 ? 'improving' : pct <= 30 ? 'deteriorating' : 'stable'
  const DeltaIcon    = pct >= 70 ? ArrowUpRight : pct <= 30 ? ArrowDownRight : Minus

  return (
    <div className="rounded-xl p-4 bg-white border border-slate-200 card-shadow space-y-2">
      <div className="flex items-start justify-between">
        <p className="text-xs font-semibold text-slate-600">{label}</p>
        {percentile != null && (
          <Badge variant={badgeVariant} className="flex-shrink-0 ml-2 text-[10px]">
            {percentile}th pctl
          </Badge>
        )}
      </div>

      <div className="flex items-end gap-4">
        <div>
          <p className="text-2xl font-bold text-slate-900 leading-none">{patientVal}</p>
          <p className="text-[10px] text-slate-400 mt-1">Patient value</p>
        </div>
        {peerMean != null && (
          <div className="pb-0.5 flex items-center gap-1">
            <DeltaIcon size={12} className={clsx(
              pct >= 70 ? 'text-emerald-500' : pct <= 30 ? 'text-rose-500' : 'text-slate-400',
            )} />
            <div>
              <p className="text-sm font-semibold text-slate-600 leading-none">{peerMean.toFixed(1)}</p>
              <p className="text-[10px] text-slate-400 mt-0.5">Peer mean</p>
            </div>
          </div>
        )}
      </div>

      {percentile != null && <PercentileBar pct={pct} />}

      {interpretation && (
        <p className="text-xs text-slate-500 leading-relaxed pt-1 border-t border-slate-50">
          {interpretation}
        </p>
      )}
    </div>
  )
}

// ── Custom radar shape ────────────────────────────────────────────────────────
const CustomRadarDot = (props: any) => {
  const { cx, cy } = props
  return <circle cx={cx} cy={cy} r={3} fill="#6366F1" stroke="#fff" strokeWidth={1.5} />
}

export default function PeersPage({ patient }: Props) {
  const r  = patient.reasoning
  const cr = patient.clinical_reasoning

  const actionable    = r.top_actionable_factors
  const nonActionable = r.top_non_actionable_factors

  const barData = actionable
    .filter(f => f.percentile != null && f.phenotype_mean != null)
    .map(f => ({
      name:     f.display_name.length > 18 ? f.display_name.slice(0, 16) + '…' : f.display_name,
      patient:  typeof f.patient_value === 'number' ? +f.patient_value.toFixed(2) : f.patient_value,
      peerMean: +(f.phenotype_mean!.toFixed(2)),
    }))

  const radarData = actionable
    .filter(f => f.percentile != null)
    .slice(0, 7)
    .map(f => ({
      subject:    f.display_name.length > 14 ? f.display_name.slice(0, 12) + '…' : f.display_name,
      percentile: f.percentile!,
      fullMark:   100,
    }))

  // Summary stats from peer_stats
  const ps = patient.peer_stats

  return (
    <div className="space-y-5 animate-fade-in">

      {/* ── Header ── */}
      <Card>
        <div className="flex items-start justify-between gap-4">
          <div className="flex-1">
            <div className="flex items-center gap-2 mb-2">
              <Users size={18} className="text-indigo-500" />
              <h2 className="text-base font-semibold text-slate-800">Peer Comparison</h2>
              <Badge variant="indigo">{r.n_peers} peers</Badge>
            </div>
            <p className="text-sm text-slate-600 leading-relaxed">{cr.peer_summary}</p>
          </div>

          {/* Peer outcome summary pills */}
          {ps && (
            <div className="flex-shrink-0 grid grid-cols-3 gap-2 text-center">
              {[
                { label: 'Improving',     pct: ps['Improving_%'],     color: '#10B981', icon: TrendingUp },
                { label: 'Stable',        pct: ps['Stable_%'],        color: '#F59E0B', icon: Minus },
                { label: 'Deteriorating', pct: ps['Deteriorating_%'], color: '#F43F5E', icon: TrendingDown },
              ].map(({ label, pct, color, icon: Icon }) => (
                <div key={label} className="rounded-xl px-3 py-2 bg-slate-50 border border-slate-100">
                  <Icon size={14} style={{ color }} className="mx-auto mb-1" />
                  <p className="text-sm font-bold" style={{ color }}>
                    {pct != null ? `${pct.toFixed(0)}%` : 'N/A'}
                  </p>
                  <p className="text-[10px] text-slate-400">{label}</p>
                </div>
              ))}
            </div>
          )}
        </div>
      </Card>

      {/* ── Charts ── */}
      <div className="grid grid-cols-2 gap-4">

        {/* Radar chart */}
        {radarData.length >= 3 && (
          <Card>
            <CardHeader>
              <CardTitle>Percentile Radar</CardTitle>
              <span className="text-xs text-slate-400">vs. phenotype peers</span>
            </CardHeader>
            <div className="h-60">
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart data={radarData} margin={{ top: 10, right: 30, left: 30, bottom: 10 }}>
                  <PolarGrid stroke="#E2E8F0" />
                  <PolarAngleAxis dataKey="subject" tick={{ fontSize: 10, fill: '#64748B' }} />
                  <Radar
                    name="Percentile"
                    dataKey="percentile"
                    stroke="#6366F1"
                    strokeWidth={2}
                    fill="#6366F1"
                    fillOpacity={0.15}
                    dot={<CustomRadarDot />}
                  />
                  <Tooltip
                    formatter={(v) => [`${v}th percentile`, 'Rank']}
                    contentStyle={{ fontSize: 11, borderRadius: 10, border: '1px solid #E2E8F0' }}
                  />
                </RadarChart>
              </ResponsiveContainer>
            </div>
            <p className="text-[11px] text-slate-400 text-center mt-1">
              Higher percentile = ranked higher among peers for that feature
            </p>
          </Card>
        )}

        {/* Bar: patient vs peer mean */}
        {barData.length > 0 && (
          <Card>
            <CardHeader>
              <CardTitle>Patient vs Peer Mean</CardTitle>
              <span className="text-xs text-slate-400">Actionable features</span>
            </CardHeader>
            <div className="h-60">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={barData} margin={{ top: 4, right: 8, left: -12, bottom: 4 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" vertical={false} />
                  <XAxis
                    dataKey="name"
                    tick={{ fontSize: 9, fill: '#94A3B8' }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <YAxis
                    tick={{ fontSize: 10, fill: '#94A3B8' }}
                    axisLine={false}
                    tickLine={false}
                  />
                  <Tooltip
                    contentStyle={{ fontSize: 11, borderRadius: 10, border: '1px solid #E2E8F0' }}
                  />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Bar dataKey="patient"  name="Patient"   radius={[4,4,0,0]} fill="#6366F1" />
                  <Bar dataKey="peerMean" name="Peer Mean" radius={[4,4,0,0]} fill="#CBD5E1" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </Card>
        )}
      </div>

      {/* ── Actionable factors — detailed cards ── */}
      {actionable.length > 0 && (
        <div>
          <div className="flex items-center gap-2 mb-3">
            <TrendingUp size={14} className="text-indigo-500" />
            <h3 className="text-sm font-semibold text-slate-700">Actionable Factors — Patient vs Peers</h3>
          </div>
          <div className="grid grid-cols-2 gap-3">
            {actionable.map((f, i) => (
              <StatCard
                key={i}
                label={f.display_name}
                patientVal={typeof f.patient_value === 'number' ? f.patient_value.toFixed(2) : f.patient_value}
                peerMean={f.phenotype_mean}
                percentile={f.percentile}
                interpretation={f.interpretation}
              />
            ))}
          </div>
        </div>
      )}

      {/* ── Non-actionable (fixed) factors ── */}
      {nonActionable.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle>Fixed Characteristics</CardTitle>
            <span className="text-xs text-slate-400">Context only — not actionable</span>
          </CardHeader>
          <div className="grid grid-cols-2 gap-3">
            {nonActionable.map((f, i) => (
              <StatCard
                key={i}
                label={f.display_name}
                patientVal={typeof f.patient_value === 'number'
                  ? f.patient_value.toFixed(2)
                  : f.patient_value}
                peerMean={f.phenotype_mean}
                percentile={f.percentile}
                interpretation={f.interpretation}
              />
            ))}
          </div>
        </Card>
      )}

      {/* ── Interpretation cautions ── */}
      {r.cautions.length > 0 && (
        <div className="rounded-2xl bg-amber-50 border border-amber-200 p-4">
          <div className="flex items-center gap-2 mb-2.5">
            <AlertTriangle size={14} className="text-amber-600" />
            <p className="text-xs font-semibold text-amber-700">Interpretation Notes</p>
          </div>
          <ul className="space-y-1.5">
            {r.cautions.map((c, i) => (
              <li key={i} className="text-xs text-amber-700 flex gap-2">
                <span className="flex-shrink-0 mt-0.5">•</span>
                <span>{c}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  )
}
