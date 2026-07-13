import { useState, useCallback } from 'react'
import {
  LayoutDashboard,
  MessageSquare,
  FlaskConical,
  BarChart3,
  Dna,
  Users,
  Activity,
  TrendingUp,
  TrendingDown,
  Minus,
  Wifi,
} from 'lucide-react'
import Sidebar from './components/layout/Sidebar'
import Tabs, { type TabDef } from './components/ui/Tabs'
import { SkeletonCard } from './components/ui/Spinner'
import Badge from './components/ui/Badge'
import OverviewPage    from './pages/OverviewPage'
import ChatPage        from './pages/ChatPage'
import SimulationPage  from './pages/SimulationPage'
import SHAPPage        from './pages/SHAPPage'
import PhenotypePage   from './pages/PhenotypePage'
import PeersPage       from './pages/PeersPage'
import { usePatient }  from './hooks/usePatient'
import { clsx }        from './utils'

type Tab = 'overview' | 'chat' | 'simulation' | 'shap' | 'phenotype' | 'peers'

const TABS: TabDef[] = [
  { id: 'overview',   label: 'Overview',        icon: <LayoutDashboard size={13} /> },
  { id: 'chat',       label: 'AI Assistant',    icon: <MessageSquare size={13} />   },
  { id: 'simulation', label: 'Simulation',      icon: <FlaskConical size={13} />    },
  { id: 'shap',       label: 'Explainability',  icon: <BarChart3 size={13} />       },
  { id: 'phenotype',  label: 'Phenotype',       icon: <Dna size={13} />             },
  { id: 'peers',      label: 'Peer Comparison', icon: <Users size={13} />           },
]

function TrajectoryIcon({ label }: { label: string }) {
  if (label === 'Improving')     return <TrendingUp  size={11} />
  if (label === 'Deteriorating') return <TrendingDown size={11} />
  return <Minus size={11} />
}

function trajectoryVariant(label: string): 'improving' | 'stable' | 'deteriorating' {
  if (label === 'Improving')     return 'improving'
  if (label === 'Deteriorating') return 'deteriorating'
  return 'stable'
}

export default function App() {
  const [activeTab, setActiveTab] = useState<Tab>('overview')
  const { data: patient, loading, error, loadPatient } = usePatient()

  const handleLoad = useCallback(async (id: string) => {
    const p = await loadPatient(id)
    if (p) setActiveTab('overview')
  }, [loadPatient])

  const handleTabChange = useCallback((tab: string) => {
    setActiveTab(tab as Tab)
  }, [])

  const visibleTabs = TABS.filter(t => {
    if (t.id === 'shap') return true
    return !!patient
  })

  const showTabs = patient || activeTab === 'shap'

  return (
    <div className="flex h-screen overflow-hidden bg-slate-50">
      <Sidebar
        patient={patient}
        activeTab={activeTab}
        onTabChange={handleTabChange}
        onLoadPatient={handleLoad}
        loading={loading}
      />

      {/* ── Main content ── */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">

        {/* ── Top bar ── */}
        <header className="bg-white shrink-0 flex items-center justify-between px-6 h-12 border-b border-slate-200">
          {/* Breadcrumb / Patient header */}
          <div className="flex items-center gap-2 min-w-0">
            {patient ? (
              <>
                <span className="text-xs text-slate-400">Patient</span>
                <span className="text-slate-300">/</span>
                <span className="text-sm font-semibold text-slate-800 truncate max-w-[160px]">
                  {patient.patient_id}
                </span>
                <Badge variant={trajectoryVariant(patient.predicted_label)} dot>
                  <TrajectoryIcon label={patient.predicted_label} />
                  {patient.predicted_label}
                </Badge>
                {patient.predicted_score != null && (
                  <span className="text-xs text-slate-500 font-mono hidden sm:inline">
                    {patient.predicted_score.toFixed(1)} pts
                  </span>
                )}
                <span className="text-xs text-slate-400 hidden md:inline truncate max-w-[200px]">
                  {patient.phenotype}
                </span>
              </>
            ) : (
              <div className="flex items-center gap-2">
                <Activity size={14} className="text-indigo-500" />
                <span className="text-sm font-semibold text-slate-700">CognixAI</span>
                <span className="text-xs text-slate-400 hidden sm:inline">— Clinical Decision Support</span>
              </div>
            )}
          </div>

          {/* Status indicator */}
          <div className="flex items-center gap-2 flex-shrink-0">
            <Wifi size={12} className="text-emerald-500" />
            <span className="text-xs text-slate-500">Connected</span>
          </div>
        </header>

        {/* ── Tab bar ── */}
        {showTabs && (
          <div className="bg-white shrink-0 border-b border-slate-200 px-6">
            <Tabs tabs={visibleTabs} active={activeTab} onChange={handleTabChange} />
          </div>
        )}

        {/* ── Page content ── */}
        <main className="flex-1 overflow-hidden flex flex-col">

          {/* Non-chat pages (scrollable, hidden when on chat tab) */}
          <div className={clsx(
            'flex-1 overflow-auto p-6',
            activeTab === 'chat' && 'hidden',
          )}>
            {loading ? (
              <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-3 gap-4">
                {[...Array(6)].map((_, i) => <SkeletonCard key={i} lines={4} />)}
              </div>
            ) : error ? (
              <div className="flex items-center justify-center h-full">
                <div className="text-center max-w-sm">
                  <div className="w-14 h-14 rounded-2xl bg-rose-50 flex items-center justify-center mx-auto mb-4">
                    <Activity size={24} className="text-rose-400" />
                  </div>
                  <h3 className="text-base font-semibold text-slate-800 mb-1">Patient not found</h3>
                  <p className="text-sm text-slate-500 mb-4">{error}</p>
                  <p className="text-xs text-slate-400">
                    Try a different ID (e.g. PID0000, PID0001…)
                  </p>
                </div>
              </div>
            ) : !patient && activeTab !== 'shap' ? (
              <WelcomeScreen />
            ) : (
              <>
                {activeTab === 'overview'   && patient && <OverviewPage   patient={patient} />}
                {activeTab === 'simulation' && patient && <SimulationPage patient={patient} />}
                {activeTab === 'shap'                  && <SHAPPage       patient={patient!} />}
                {activeTab === 'phenotype'  && patient && <PhenotypePage  patient={patient} />}
                {activeTab === 'peers'      && patient && <PeersPage      patient={patient} />}
              </>
            )}
          </div>

          {/* Chat — always mounted once a patient is loaded so history persists across tab switches.
              Starts loading the intro in the background immediately when patient changes. */}
          {patient && (
            <div className={clsx(
              'flex-1 flex flex-col min-h-0 overflow-hidden',
              activeTab !== 'chat' && 'hidden',
            )}>
              <ChatPage patient={patient} />
            </div>
          )}
        </main>
      </div>
    </div>
  )
}

// ── Welcome / landing screen ──────────────────────────────────────────────────
function WelcomeScreen() {
  const features = [
    {
      icon: <LayoutDashboard size={18} className="text-indigo-500" />,
      gradient: 'from-indigo-50 to-violet-50',
      border: 'border-indigo-100',
      title: 'Predictive Insights',
      desc: 'XGBoost predictions with trajectory labels — Improving, Stable, or Deteriorating.',
    },
    {
      icon: <BarChart3 size={18} className="text-emerald-500" />,
      gradient: 'from-emerald-50 to-sky-50',
      border: 'border-emerald-100',
      title: 'SHAP Explainability',
      desc: 'Interactive SHAP visualisations reveal exactly which features drive each prediction.',
    },
    {
      icon: <FlaskConical size={18} className="text-violet-500" />,
      gradient: 'from-violet-50 to-indigo-50',
      border: 'border-violet-100',
      title: 'What-If Simulation',
      desc: 'Model the impact of clinical interventions before applying them.',
    },
    {
      icon: <MessageSquare size={18} className="text-sky-500" />,
      gradient: 'from-sky-50 to-indigo-50',
      border: 'border-sky-100',
      title: 'AI Chat Assistant',
      desc: 'Ask clinical questions about any patient and get streaming LLM responses.',
    },
    {
      icon: <Dna size={18} className="text-amber-500" />,
      gradient: 'from-amber-50 to-orange-50',
      border: 'border-amber-100',
      title: 'Phenotype Analysis',
      desc: 'Cluster-based patient profiles with outcome distributions and characteristics.',
    },
    {
      icon: <Users size={18} className="text-rose-500" />,
      gradient: 'from-rose-50 to-pink-50',
      border: 'border-rose-100',
      title: 'Peer Benchmarking',
      desc: 'Compare patients against their phenotype cohort with radar charts and percentiles.',
    },
  ]

  return (
    <div className="flex flex-col items-center justify-center min-h-full text-center px-6 py-16 animate-fade-in">
      {/* Hero */}
      <div className="w-16 h-16 rounded-2xl gradient-brand flex items-center justify-center shadow-lg shadow-indigo-300/40 mb-6">
        <Activity size={28} className="text-white" />
      </div>
      <h1 className="text-2xl font-bold text-slate-900 tracking-tight mb-2">
        Welcome to <span className="text-gradient">CognixAI</span>
      </h1>
      <p className="text-sm text-slate-500 max-w-md leading-relaxed mb-10">
        A clinical decision support platform powered by XGBoost + SHAP.
        Search for a patient in the sidebar to get started.
      </p>

      {/* Feature grid */}
      <div className="grid grid-cols-2 md:grid-cols-3 gap-3 max-w-2xl w-full">
        {features.map((f) => (
          <div
            key={f.title}
            className={clsx(
              'rounded-2xl p-4 text-left border bg-gradient-to-br',
              f.gradient,
              f.border,
              'transition-all duration-200 hover:-translate-y-0.5 hover:shadow-md',
            )}
          >
            <div className="mb-3 w-8 h-8 rounded-xl bg-white shadow-sm flex items-center justify-center">
              {f.icon}
            </div>
            <h3 className="text-sm font-semibold text-slate-800 mb-1">{f.title}</h3>
            <p className="text-xs text-slate-500 leading-relaxed">{f.desc}</p>
          </div>
        ))}
      </div>

      <p className="text-xs text-slate-400 mt-10">
        Enter a patient ID in the sidebar (e.g.{' '}
        <span className="font-mono font-medium text-slate-500">PID0000</span>)
      </p>
    </div>
  )
}
