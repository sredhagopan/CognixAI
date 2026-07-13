import { useState, useEffect, useRef, type FormEvent } from 'react'
import {
  LayoutDashboard,
  MessageSquare,
  FlaskConical,
  BarChart3,
  Dna,
  Users,
  Search,
  ChevronRight,
  Activity,
  Loader2,
  X,
  Cpu,
} from 'lucide-react'
import { fetchPatients } from '../../services/api'
import { clsx } from '../../utils'
import type { PatientData } from '../../types'

type Tab = 'overview' | 'chat' | 'simulation' | 'shap' | 'phenotype' | 'peers'

interface SidebarProps {
  patient: PatientData | null
  activeTab: Tab
  onTabChange: (tab: Tab) => void
  onLoadPatient: (id: string) => void
  loading: boolean
}

const NAV_ITEMS: {
  id: Tab
  label: string
  icon: React.ComponentType<{ size?: number; className?: string }>
  requiresPatient: boolean
  description: string
}[] = [
  { id: 'overview',   label: 'Overview',        icon: LayoutDashboard, requiresPatient: true,  description: 'Patient summary' },
  { id: 'chat',       label: 'AI Assistant',    icon: MessageSquare,   requiresPatient: true,  description: 'Clinical Q&A' },
  { id: 'simulation', label: 'Simulation',      icon: FlaskConical,    requiresPatient: true,  description: 'What-if analysis' },
  { id: 'shap',       label: 'Explainability',  icon: BarChart3,       requiresPatient: false, description: 'SHAP feature drivers' },
  { id: 'phenotype',  label: 'Phenotype',       icon: Dna,             requiresPatient: true,  description: 'Cluster analysis' },
  { id: 'peers',      label: 'Peer Comparison', icon: Users,           requiresPatient: true,  description: 'Cohort benchmarks' },
]

export default function Sidebar({
  patient,
  activeTab,
  onTabChange,
  onLoadPatient,
  loading,
}: SidebarProps) {
  const [allIds, setAllIds]       = useState<string[]>([])
  const [suggestions, setSugs]    = useState<string[]>([])
  const [showSugs, setShowSugs]   = useState(false)
  const [inputVal, setInputVal]   = useState('')
  const inputRef                  = useRef<HTMLInputElement>(null)
  const containerRef              = useRef<HTMLDivElement>(null)

  useEffect(() => {
    fetchPatients().then(setAllIds).catch(() => {})
  }, [])

  useEffect(() => {
    if (!inputVal) { setSugs([]); return }
    const q = inputVal.toUpperCase()
    setSugs(allIds.filter(id => id.toUpperCase().includes(q)).slice(0, 8))
  }, [inputVal, allIds])

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node))
        setShowSugs(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  function handleSubmit(e: FormEvent) {
    e.preventDefault()
    const id = inputVal.trim()
    if (!id) return
    setShowSugs(false)
    onLoadPatient(id)
  }

  function handleSelect(id: string) {
    setInputVal(id)
    setShowSugs(false)
    onLoadPatient(id)
  }

  function handleClear() {
    setInputVal('')
    setSugs([])
    inputRef.current?.focus()
  }

  return (
    <aside className="w-64 shrink-0 flex flex-col h-full overflow-hidden bg-slate-900">
      {/* ── Logo ── */}
      <div className="px-5 py-5 border-b border-white/[0.06]">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-xl gradient-brand flex items-center justify-center flex-shrink-0 shadow-lg shadow-indigo-900/50">
            <Activity size={16} className="text-white" />
          </div>
          <div>
            <p className="text-[13px] font-bold text-white tracking-tight leading-none">CognixAI</p>
            <p className="text-[10px] text-slate-500 mt-0.5 leading-none font-medium tracking-wide uppercase">
              Clinical Intelligence
            </p>
          </div>
        </div>
      </div>

      {/* ── Patient search ── */}
      <div className="px-4 py-4 border-b border-white/[0.06]">
        <p className="text-[10px] font-semibold uppercase tracking-widest text-slate-500 mb-2.5 px-0.5">
          Patient Lookup
        </p>
        <form onSubmit={handleSubmit}>
          <div className="relative" ref={containerRef}>
            <Search
              size={13}
              className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500 pointer-events-none"
            />
            <input
              ref={inputRef}
              value={inputVal}
              onChange={(e) => { setInputVal(e.target.value); setShowSugs(true) }}
              onFocus={() => { if (inputVal) setShowSugs(true) }}
              placeholder="Search patient ID…"
              className="w-full rounded-xl py-2 pl-8 pr-8 text-[13px] text-slate-200 bg-white/[0.06]
                         border border-white/[0.08] placeholder:text-slate-600
                         focus:outline-none focus:border-indigo-500/50 focus:bg-white/[0.08]
                         transition-all duration-200"
            />
            {loading ? (
              <Loader2
                size={13}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-indigo-400 animate-spin"
              />
            ) : inputVal ? (
              <button
                type="button"
                onClick={handleClear}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-500 hover:text-slate-300 transition-colors"
              >
                <X size={12} />
              </button>
            ) : null}

            {showSugs && suggestions.length > 0 && (
              <div
                className="absolute top-full left-0 right-0 mt-1.5 rounded-xl overflow-hidden z-50 bg-slate-800
                           border border-white/[0.08] shadow-xl shadow-black/30"
              >
                {suggestions.map((id) => (
                  <button
                    key={id}
                    type="button"
                    onClick={() => handleSelect(id)}
                    className="w-full text-left px-3 py-2 text-[13px] text-slate-300
                               hover:bg-white/[0.08] transition-colors flex items-center justify-between"
                  >
                    <span>{id}</span>
                    <ChevronRight size={12} className="text-slate-600" />
                  </button>
                ))}
              </div>
            )}
          </div>
        </form>

        {/* Active patient badge */}
        {patient && (
          <div className="mt-3 px-3 py-2.5 rounded-xl bg-indigo-600/10 border border-indigo-500/20">
            <div className="flex items-center justify-between">
              <div className="min-w-0">
                <p className="text-[10px] font-semibold text-indigo-400 uppercase tracking-wide">Active Patient</p>
                <p className="text-sm font-bold text-white mt-0.5 truncate">{patient.patient_id}</p>
              </div>
              <div className="w-2 h-2 rounded-full bg-emerald-500 flex-shrink-0 animate-pulse-ring" />
            </div>
            <p className="text-[11px] text-slate-400 mt-1 truncate leading-tight">{patient.phenotype}</p>
          </div>
        )}
      </div>

      {/* ── Navigation ── */}
      <nav className="flex-1 overflow-y-auto py-3 px-3">
        <p className="text-[10px] font-semibold uppercase tracking-widest text-slate-600 mb-2 px-1.5">
          Navigation
        </p>
        <div className="space-y-0.5">
          {NAV_ITEMS.map((item) => {
            const Icon     = item.icon
            const disabled = item.requiresPatient && !patient
            const isActive = activeTab === item.id

            return (
              <button
                key={item.id}
                onClick={() => !disabled && onTabChange(item.id)}
                disabled={disabled}
                title={disabled ? 'Load a patient first' : item.description}
                className={clsx(
                  'w-full flex items-center gap-3 px-3 py-2.5 rounded-xl text-[13px]',
                  'transition-all duration-150 text-left group',
                  isActive
                    ? 'bg-indigo-600/15 text-indigo-300 font-medium border border-indigo-500/20'
                    : disabled
                      ? 'text-slate-700 cursor-not-allowed'
                      : 'text-slate-400 hover:bg-white/[0.05] hover:text-slate-200',
                )}
              >
                <Icon
                  size={15}
                  className={clsx(
                    'flex-shrink-0 transition-colors',
                    isActive   ? 'text-indigo-400' :
                    disabled   ? 'text-slate-700'  :
                    'text-slate-500 group-hover:text-slate-300',
                  )}
                />
                <span className="truncate">{item.label}</span>
                {isActive && (
                  <span className="ml-auto w-1.5 h-1.5 rounded-full bg-indigo-500 flex-shrink-0" />
                )}
              </button>
            )
          })}
        </div>
      </nav>

      {/* ── Footer ── */}
      <div className="px-4 py-4 border-t border-white/[0.06]">
        <div className="flex items-center gap-2">
          <Cpu size={12} className="text-slate-600 flex-shrink-0" />
          <p className="text-[11px] text-slate-600 truncate">
            {allIds.length > 0 ? `${allIds.length} patients · ` : ''}XGBoost + SHAP
          </p>
        </div>
      </div>
    </aside>
  )
}
