import { Dna, Users, Target, BookOpen, Lightbulb } from 'lucide-react'
import type { PatientData } from '../types'
import Card, { CardHeader, CardTitle } from '../components/ui/Card'
import Badge from '../components/ui/Badge'
import {
  PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer,
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
} from 'recharts'

interface Props { patient: PatientData }

const TRAJ_COLORS = {
  Improving:     '#10B981',
  Stable:        '#F59E0B',
  Deteriorating: '#F43F5E',
}

const ALL_PHENOTYPES = [
  {
    name:          'Cardiometabolic / Cholesterol Dominant',
    n:             245,
    pct:           49.0,
    improving_pct: 38.8,
    stable_pct:    19.6,
    det_pct:       41.6,
  },
  {
    name:          'Cardiometabolic / BMI Dominant',
    n:             162,
    pct:           32.4,
    improving_pct: 47.5,
    stable_pct:    17.9,
    det_pct:       34.6,
  },
  {
    name:          'Cardiometabolic / Heart Rate Dominant',
    n:             93,
    pct:           18.6,
    improving_pct: 37.6,
    stable_pct:    22.6,
    det_pct:       39.8,
  },
]

function OutcomeBar({ improving, stable, deteriorating }: {
  improving: number
  stable: number
  deteriorating: number
}) {
  return (
    <div className="h-3 rounded-full overflow-hidden flex gap-0.5">
      <div style={{ width: `${improving}%`, backgroundColor: TRAJ_COLORS.Improving }} className="rounded-l-full" />
      <div style={{ width: `${stable}%`, backgroundColor: TRAJ_COLORS.Stable }} />
      <div style={{ width: `${deteriorating}%`, backgroundColor: TRAJ_COLORS.Deteriorating }} className="rounded-r-full" />
    </div>
  )
}

function QABlock({ q, a }: { q: string; a: string }) {
  return (
    <div className="border-b border-slate-100 pb-4 last:border-0 last:pb-0">
      <p className="text-xs font-semibold text-indigo-600 mb-1.5 flex items-center gap-1">
        <Lightbulb size={11} /> {q}
      </p>
      <p className="text-sm text-slate-700 leading-relaxed">{a}</p>
    </div>
  )
}

export default function PhenotypePage({ patient }: Props) {
  const rag = patient.rag_info

  const outcomeData = [
    { name: 'Improving',     value: patient.improving_pct,     fill: TRAJ_COLORS.Improving     },
    { name: 'Stable',        value: patient.stable_pct,        fill: TRAJ_COLORS.Stable        },
    { name: 'Deteriorating', value: patient.deteriorating_pct, fill: TRAJ_COLORS.Deteriorating },
  ]

  const chatResponses = rag.chatbot_responses || {}

  const barData = ALL_PHENOTYPES.map(p => ({
    name:             p.name.replace('Cardiometabolic / ', '').replace(' Dominant', ''),
    'Improving %':    p.improving_pct,
    'Stable %':       p.stable_pct,
    'Deteriorating %':p.det_pct,
    isCurrent:        p.name === patient.phenotype,
  }))

  return (
    <div className="space-y-5 animate-fade-in">

      {/* ── Phenotype hero card ── */}
      <Card className="relative overflow-hidden">
        <div className="absolute top-0 left-0 right-0 h-1 rounded-t-2xl bg-gradient-to-r from-violet-500 to-indigo-500" />
        <div className="flex items-start gap-5">
          <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-violet-50 to-indigo-50
                          border border-violet-100 flex items-center justify-center flex-shrink-0">
            <Dna size={24} className="text-violet-500" />
          </div>
          <div className="flex-1 min-w-0">
            <div className="flex items-start justify-between gap-3 mb-2">
              <div>
                <h2 className="text-base font-bold text-slate-900 leading-tight">{patient.phenotype}</h2>
                <div className="flex items-center gap-3 mt-1.5 flex-wrap">
                  <Badge variant="purple">{rag.prevalence_pct}% of cohort</Badge>
                  <Badge variant="indigo" dot>{patient.n_patients} patients</Badge>
                  <span className="text-xs text-slate-500">
                    Domain: <span className="font-medium">{rag.domain || 'Cardiometabolic'}</span>
                  </span>
                  <span className="text-xs text-slate-500">
                    Driver: <span className="font-medium">{rag.dominant_feature || patient.dominant_feature}</span>
                  </span>
                </div>
              </div>
            </div>
            <p className="text-sm text-slate-600 leading-relaxed">{patient.phenotype_summary}</p>
          </div>
        </div>

        {/* Outcome bar */}
        <div className="mt-5 pt-4 border-t border-slate-100">
          <div className="flex items-center justify-between mb-2">
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide">Outcome Distribution</p>
            <div className="flex items-center gap-4">
              {outcomeData.map(d => (
                <div key={d.name} className="flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full" style={{ backgroundColor: d.fill }} />
                  <span className="text-[11px] text-slate-500">{d.name} {d.value.toFixed(0)}%</span>
                </div>
              ))}
            </div>
          </div>
          <OutcomeBar
            improving={patient.improving_pct}
            stable={patient.stable_pct}
            deteriorating={patient.deteriorating_pct}
          />
        </div>
      </Card>

      {/* ── Charts row ── */}
      <div className="grid grid-cols-2 gap-4">

        {/* Outcome donut chart */}
        <Card>
          <CardHeader>
            <CardTitle icon={<Target size={14} />}>Outcome Distribution</CardTitle>
            <span className="text-xs text-slate-400">{patient.n_patients} pts</span>
          </CardHeader>
          <div className="h-52">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={outcomeData}
                  cx="50%" cy="50%"
                  innerRadius={55}
                  outerRadius={85}
                  dataKey="value"
                  startAngle={90}
                  endAngle={-270}
                  paddingAngle={2}
                >
                  {outcomeData.map((d, i) => (
                    <Cell key={i} fill={d.fill} strokeWidth={0} />
                  ))}
                </Pie>
                <Tooltip
                  formatter={(v) => [`${Number(v).toFixed(1)}%`, '']}
                  contentStyle={{ fontSize: 11, borderRadius: 10, border: '1px solid #E2E8F0' }}
                />
                <Legend
                  formatter={(value, entry: any) => (
                    <span className="text-xs text-slate-600">
                      {value}: <strong>{entry.payload.value.toFixed(1)}%</strong>
                    </span>
                  )}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="grid grid-cols-3 gap-2 mt-1">
            {outcomeData.map(d => (
              <div key={d.name} className="text-center rounded-xl p-2.5 bg-slate-50">
                <p className="text-xl font-bold" style={{ color: d.fill }}>{d.value.toFixed(0)}%</p>
                <p className="text-[10px] text-slate-500 mt-0.5">{d.name}</p>
              </div>
            ))}
          </div>
        </Card>

        {/* Characteristics & recommendation */}
        <Card>
          <CardHeader>
            <CardTitle icon={<BookOpen size={14} />}>Characteristics</CardTitle>
          </CardHeader>
          <div className="space-y-4">
            <div>
              <p className="text-[11px] font-semibold text-slate-400 uppercase tracking-wide mb-1.5">
                Key clinical factors
              </p>
              <p className="text-sm text-slate-700 leading-relaxed">{patient.characteristics}</p>
            </div>

            {rag.recommendation && (
              <div className="pt-3 border-t border-slate-100">
                <p className="text-[11px] font-semibold text-slate-400 uppercase tracking-wide mb-1.5">
                  Clinical recommendation
                </p>
                <p className="text-sm text-slate-700 leading-relaxed">{rag.recommendation}</p>
              </div>
            )}

            {rag.evidence_context && (
              <div className="pt-3 border-t border-slate-100 rounded-xl bg-indigo-50/50 p-3 -mx-1">
                <p className="text-[11px] font-semibold text-indigo-600 uppercase tracking-wide mb-1">
                  Evidence context
                </p>
                <p className="text-xs text-slate-600 leading-relaxed italic">{rag.evidence_context}</p>
              </div>
            )}
          </div>
        </Card>
      </div>

      {/* ── Phenotype Q&A insights ── */}
      {Object.keys(chatResponses).length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle icon={<Lightbulb size={14} />}>Phenotype Insights</CardTitle>
            <Badge variant="indigo">RAG-powered</Badge>
          </CardHeader>
          <div className="space-y-4">
            {chatResponses.which_cluster && (
              <QABlock q="Which phenotype cluster is this?" a={chatResponses.which_cluster} />
            )}
            {chatResponses.what_defines && (
              <QABlock q="What defines this phenotype?" a={chatResponses.what_defines} />
            )}
            {chatResponses.what_to_focus && (
              <QABlock q="What should this patient focus on?" a={chatResponses.what_to_focus} />
            )}
            {chatResponses.what_improves && (
              <QABlock q="What improves outcomes in this group?" a={chatResponses.what_improves} />
            )}
          </div>
        </Card>
      )}

      {/* ── All phenotypes stacked bar ── */}
      <Card>
        <CardHeader>
          <CardTitle icon={<Users size={14} />}>All Phenotype Groups — Outcome Comparison</CardTitle>
        </CardHeader>
        <div className="h-56 mb-2">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={barData} margin={{ top: 4, right: 16, left: -8, bottom: 4 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#F1F5F9" />
              <XAxis
                dataKey="name"
                tick={{ fontSize: 11, fill: '#64748B' }}
                axisLine={false}
                tickLine={false}
              />
              <YAxis
                tick={{ fontSize: 10, fill: '#94A3B8' }}
                axisLine={false}
                tickLine={false}
                tickFormatter={v => `${v}%`}
              />
              <Tooltip
                formatter={(v) => [`${Number(v).toFixed(1)}%`, '']}
                contentStyle={{ fontSize: 11, borderRadius: 10, border: '1px solid #E2E8F0' }}
              />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Bar dataKey="Improving %"     stackId="a" fill={TRAJ_COLORS.Improving}     radius={[0,0,0,0]} />
              <Bar dataKey="Stable %"        stackId="a" fill={TRAJ_COLORS.Stable}        />
              <Bar dataKey="Deteriorating %" stackId="a" fill={TRAJ_COLORS.Deteriorating} radius={[4,4,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <p className="text-xs text-slate-400">
          Current patient phenotype:{' '}
          <span className="font-semibold text-indigo-600">{patient.phenotype}</span>
        </p>
      </Card>
    </div>
  )
}

