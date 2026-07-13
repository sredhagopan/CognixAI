import {
  useState, useEffect, useRef, useCallback, type KeyboardEvent,
} from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import {
  Send, Square, Copy, Check, Activity, Bot, User,
  TrendingUp, TrendingDown, Minus, ArrowUpRight, ArrowDownRight,
  FlaskConical,
} from 'lucide-react'
import { streamIntro, streamChat } from '../services/api'
import type { PatientData, ChatMessage, SimulationResult } from '../types'
import { uid, clsx, trajectoryFill, fmtDelta, featureDisplayName } from '../utils'

// Module-level cache: intro messages + history per patient so revisiting a patient
// is instant (no second LLM call) and switching tabs never replays the briefing.
const _introCache = new Map<string, {
  messages: ChatMessage[]
  history:  { role: 'user' | 'assistant'; content: string }[]
}>()

const SUGGESTED_QUESTIONS = [
  'Why did the model predict this outcome?',
  'What are the biggest risk factors?',
  'What should be improved first?',
  'Compare this patient with similar cases.',
  'Generate a clinical summary.',
  'What if cholesterol dropped to 180?',
  "What's the best single intervention?",
  'What combination of changes would help most?',
]

interface Props { patient: PatientData }

// ── Copy button ───────────────────────────────────────────────────────────────
function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false)

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch { /* ignore */ }
  }

  return (
    <button
      onClick={handleCopy}
      className="text-slate-400 hover:text-slate-600 transition-colors p-1 rounded-md hover:bg-slate-100"
      title={copied ? 'Copied!' : 'Copy message'}
    >
      {copied ? <Check size={13} className="text-emerald-500" /> : <Copy size={13} />}
    </button>
  )
}

// ── Simulation result card ────────────────────────────────────────────────────
function SimResultCard({ result }: { result: SimulationResult }) {
  const delta    = result.delta
  const improved = delta > 0
  const minimal  = Math.abs(delta) < 0.5

  function TrajBadge({ label }: { label: string }) {
    const color = trajectoryFill(label)
    const Icon = label === 'Improving' ? TrendingUp : label === 'Deteriorating' ? TrendingDown : Minus
    return (
      <span className="inline-flex items-center gap-1 text-xs font-medium px-2 py-0.5 rounded-full"
            style={{ backgroundColor: `${color}18`, color }}>
        <Icon size={10} /> {label}
      </span>
    )
  }

  return (
    <div className="my-3 rounded-xl border border-slate-200 bg-slate-50 overflow-hidden">
      <div className="flex items-center gap-2 px-4 py-2.5 bg-white border-b border-slate-100">
        <FlaskConical size={14} className="text-indigo-500" />
        <span className="text-xs font-semibold text-slate-700">
          Simulation —{' '}
          {result.mode === 'best_single'       ? 'Best Single Change' :
           result.mode === 'best_combination'  ? 'Best Combination'   : 'What-If Scenario'}
        </span>
      </div>

      <div className="p-3 space-y-2">
        {result.changes_applied && Object.keys(result.changes_applied).length > 0 && (
          <div className="text-xs text-slate-600 space-y-0.5">
            {Object.entries(result.changes_applied).map(([feat, val]) => (
              <div key={feat} className="flex gap-1.5">
                <span className="text-slate-400">{featureDisplayName(feat)}:</span>
                <span className="font-medium text-slate-700">
                  {result.original_values?.[feat] != null
                    ? `${result.original_values[feat]} → ${val}`
                    : val}
                </span>
              </div>
            ))}
          </div>
        )}

        <div className="grid grid-cols-3 gap-2">
          <div className="rounded-lg p-2.5 text-center bg-white border border-slate-100">
            <p className="text-[10px] text-slate-400 mb-1">Before</p>
            <p className="text-base font-bold text-slate-800">{result.before_score.toFixed(1)}</p>
            <TrajBadge label={result.before_label} />
          </div>

          <div className={clsx(
            'rounded-lg p-2.5 text-center border',
            minimal  ? 'bg-slate-50 border-slate-100'    :
            improved ? 'bg-emerald-50 border-emerald-100' : 'bg-rose-50 border-rose-100',
          )}>
            <p className="text-[10px] text-slate-400 mb-1">Change</p>
            <p className={clsx(
              'text-base font-bold flex items-center justify-center gap-0.5',
              minimal  ? 'text-slate-600'   :
              improved ? 'text-emerald-700' : 'text-rose-700',
            )}>
              {!minimal && (improved
                ? <ArrowUpRight size={14} />
                : <ArrowDownRight size={14} />
              )}
              {fmtDelta(delta)}
            </p>
            <p className="text-[10px] text-slate-400">pts</p>
          </div>

          <div className={clsx(
            'rounded-lg p-2.5 text-center border',
            result.label_changed
              ? improved ? 'bg-emerald-50 border-emerald-100' : 'bg-rose-50 border-rose-100'
              : 'bg-white border-slate-100',
          )}>
            <p className="text-[10px] text-slate-400 mb-1">After</p>
            <p className="text-base font-bold text-slate-800">{result.after_score.toFixed(1)}</p>
            <TrajBadge label={result.after_label} />
          </div>
        </div>

        {result.individual_impacts && result.individual_impacts.length > 0 && (
          <div className="pt-2 border-t border-slate-100 space-y-1">
            {result.individual_impacts.map((imp, i) => (
              <div key={i} className="flex items-center justify-between text-xs">
                <span className="text-slate-500">{featureDisplayName(imp.feature)}</span>
                <span className="font-medium text-emerald-600">+{imp.delta.toFixed(2)} pts</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

// ── Message bubble ────────────────────────────────────────────────────────────
function MessageBubble({ msg }: { msg: ChatMessage }) {
  const isUser = msg.role === 'user'

  return (
    <div className={clsx('group flex gap-3 animate-fade-in', isUser && 'flex-row-reverse')}>
      {/* Avatar */}
      <div className={clsx(
        'flex-shrink-0 w-8 h-8 rounded-xl flex items-center justify-center',
        isUser
          ? 'gradient-brand shadow-sm shadow-indigo-200'
          : 'bg-slate-100 border border-slate-200',
      )}>
        {isUser
          ? <User size={14} className="text-white" />
          : <Bot size={14} className="text-slate-500" />}
      </div>

      {/* Bubble */}
      <div className={clsx(
        'flex-1 max-w-[80%] space-y-0.5',
        isUser && 'flex flex-col items-end',
      )}>
        {msg.simResult && <SimResultCard result={msg.simResult} />}
        <div className={clsx(
          'relative rounded-2xl px-4 py-3',
          isUser
            ? 'bg-indigo-600 text-white rounded-tr-sm shadow-sm shadow-indigo-200'
            : 'bg-white border border-slate-200 rounded-tl-sm card-shadow',
        )}>
          {/* Copy button — appears on hover */}
          {!isUser && msg.content && !msg.isStreaming && (
            <div className="absolute -top-2 right-3 opacity-0 group-hover:opacity-100 transition-opacity">
              <CopyButton text={msg.content} />
            </div>
          )}

          {isUser ? (
            <p className="text-sm text-white leading-relaxed">{msg.content}</p>
          ) : (
            <div className={clsx('prose-chat', msg.isStreaming && 'typing-cursor')}>
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {msg.content || (msg.isStreaming ? '' : '')}
              </ReactMarkdown>
            </div>
          )}
        </div>
        <p className={clsx(
          'text-[10px] text-slate-400 px-1',
          isUser && 'text-right',
        )}>
          {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
        </p>
      </div>
    </div>
  )
}

// ── Typing indicator ──────────────────────────────────────────────────────────
function TypingIndicator() {
  return (
    <div className="flex gap-3 animate-fade-in">
      <div className="flex-shrink-0 w-8 h-8 rounded-xl bg-slate-100 border border-slate-200
                      flex items-center justify-center">
        <Bot size={14} className="text-slate-500" />
      </div>
      <div className="bg-white border border-slate-200 rounded-2xl rounded-tl-sm px-4 py-3 card-shadow">
        <div className="flex gap-1.5 items-center h-4">
          {[0, 1, 2].map(i => (
            <span
              key={i}
              className="w-1.5 h-1.5 bg-indigo-400 rounded-full animate-bounce"
              style={{ animationDelay: `${i * 150}ms`, animationDuration: '0.8s' }}
            />
          ))}
        </div>
      </div>
    </div>
  )
}

// ── Suggested prompt chip ─────────────────────────────────────────────────────
function PromptChip({ text, onClick }: { text: string; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className="text-xs px-3 py-1.5 rounded-xl bg-slate-50 border border-slate-200
                 text-slate-600 hover:bg-indigo-50 hover:text-indigo-700 hover:border-indigo-200
                 transition-all duration-150 text-left"
    >
      {text}
    </button>
  )
}

export default function ChatPage({ patient }: Props) {
  const [messages, setMessages]       = useState<ChatMessage[]>([])
  const [input, setInput]             = useState('')
  const [streaming, setStreaming]     = useState(false)
  const [introLoaded, setIntroLoaded] = useState(false)

  const historyRef     = useRef<{ role: 'user' | 'assistant'; content: string }[]>([])
  const abortRef       = useRef<(() => void) | null>(null)
  const bottomRef      = useRef<HTMLDivElement>(null)
  const inputRef       = useRef<HTMLTextAreaElement>(null)
  const prevPatientId  = useRef('')

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  useEffect(() => {
    if (!patient || patient.patient_id === prevPatientId.current) return
    prevPatientId.current = patient.patient_id

    // Restore from cache — no LLM call needed
    const cached = _introCache.get(patient.patient_id)
    if (cached) {
      setMessages(cached.messages)
      historyRef.current = cached.history
      setIntroLoaded(true)
      setStreaming(false)
      return
    }

    setMessages([])
    historyRef.current = []
    setIntroLoaded(false)
    setStreaming(true)

    const introMsgId = uid()
    setMessages([{
      id: introMsgId, role: 'assistant', content: '', timestamp: new Date(), isStreaming: true,
    }])

    let accumulated = ''

    const abort = streamIntro(patient.patient_id, (evt) => {
      if (evt.type === 'token') {
        accumulated += evt.content
        setMessages(prev => prev.map(m =>
          m.id === introMsgId ? { ...m, content: accumulated } : m,
        ))
      } else if (evt.type === 'metadata') {
        const history = [
          { role: 'user'      as const, content: evt.clean_content },
          { role: 'assistant' as const, content: evt.response },
        ]
        historyRef.current = history
        setMessages(prev => {
          const updated = prev.map(m =>
            m.id === introMsgId ? { ...m, isStreaming: false } : m,
          )
          _introCache.set(patient.patient_id, { messages: updated, history })
          return updated
        })
        setStreaming(false)
        setIntroLoaded(true)
      } else if (evt.type === 'done') {
        setStreaming(false)
        setIntroLoaded(true)
      } else if (evt.type === 'error') {
        setMessages(prev => prev.map(m =>
          m.id === introMsgId
            ? { ...m, content: `Error loading intro: ${evt.message}`, isStreaming: false }
            : m,
        ))
        setStreaming(false)
        setIntroLoaded(true)
      }
    })

    abortRef.current = abort
    return () => abort()
  }, [patient.patient_id])

  const sendMessage = useCallback(() => {
    const text = input.trim()
    if (!text || streaming) return
    setInput('')

    const userMsg: ChatMessage = {
      id: uid(), role: 'user', content: text, timestamp: new Date(),
    }
    setMessages(prev => [...prev, userMsg])

    const asstId = uid()
    setMessages(prev => [...prev, {
      id: asstId, role: 'assistant', content: '', timestamp: new Date(), isStreaming: true,
    }])
    setStreaming(true)

    let accumulated   = ''
    let simResultData: SimulationResult | undefined
    const currentHistory = [...historyRef.current]
    historyRef.current.push({ role: 'user', content: text })

    const abort = streamChat(patient.patient_id, text, currentHistory, (evt) => {
      if (evt.type === 'sim_result') {
        simResultData = evt.result as SimulationResult
        setMessages(prev => prev.map(m =>
          m.id === asstId ? { ...m, simResult: simResultData } : m,
        ))
      } else if (evt.type === 'token') {
        accumulated += evt.content
        setMessages(prev => prev.map(m =>
          m.id === asstId ? { ...m, content: accumulated } : m,
        ))
      } else if (evt.type === 'metadata') {
        historyRef.current[historyRef.current.length - 1].content = evt.clean_content
        historyRef.current.push({ role: 'assistant', content: evt.response })
        setMessages(prev => prev.map(m =>
          m.id === asstId ? { ...m, isStreaming: false } : m,
        ))
        setStreaming(false)
      } else if (evt.type === 'done') {
        setStreaming(false)
        setMessages(prev => prev.map(m =>
          m.id === asstId ? { ...m, isStreaming: false } : m,
        ))
      } else if (evt.type === 'error') {
        setMessages(prev => prev.map(m =>
          m.id === asstId
            ? { ...m, content: `Error: ${evt.message}`, isStreaming: false }
            : m,
        ))
        setStreaming(false)
      }
    })

    abortRef.current = abort
  }, [input, streaming, patient.patient_id])

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  function stopStreaming() {
    abortRef.current?.()
    setMessages(prev => prev.map(m => m.isStreaming ? { ...m, isStreaming: false } : m))
    setStreaming(false)
  }

  function handleSuggest(q: string) {
    setInput(q)
    inputRef.current?.focus()
  }

  const showSuggestions = introLoaded && messages.length <= 2

  return (
    <div className="flex flex-col h-full bg-slate-50/50">

      {/* ── Patient context bar ── */}
      <div className="flex-shrink-0 px-6 py-2.5 bg-white border-b border-slate-200 flex items-center gap-2">
        <Activity size={12} className="text-indigo-500 flex-shrink-0" />
        <span className="text-xs text-slate-500">
          Context: <span className="font-semibold text-slate-700">{patient.patient_id}</span>
          {' — '}{patient.phenotype}
        </span>
      </div>

      {/* ── Messages area ── */}
      <div className="flex-1 overflow-y-auto px-6 py-6 space-y-5">
        {messages.length === 0 && !streaming && (
          <div className="flex flex-col items-center justify-center h-48 text-center animate-fade-in">
            <div className="w-12 h-12 rounded-2xl bg-indigo-50 flex items-center justify-center mb-3">
              <Bot size={22} className="text-indigo-500" />
            </div>
            <p className="text-sm font-semibold text-slate-700">Loading AI Assistant…</p>
            <p className="text-xs text-slate-400 mt-1">Preparing patient briefing for {patient.patient_id}</p>
          </div>
        )}

        {messages.map((msg) => (
          <MessageBubble key={msg.id} msg={msg} />
        ))}

        {streaming && messages.length === 0 && <TypingIndicator />}

        <div ref={bottomRef} />
      </div>

      {/* ── Suggested prompts ── */}
      {showSuggestions && (
        <div className="flex-shrink-0 px-6 pb-3">
          <p className="text-[11px] font-semibold text-slate-400 uppercase tracking-wide mb-2">
            Suggested questions
          </p>
          <div className="flex flex-wrap gap-2">
            {SUGGESTED_QUESTIONS.map((q) => (
              <PromptChip key={q} text={q} onClick={() => handleSuggest(q)} />
            ))}
          </div>
        </div>
      )}

      {/* ── Input bar ── */}
      <div className="flex-shrink-0 bg-white border-t border-slate-200 px-6 py-4">
        <div className="flex gap-3 items-end max-w-4xl mx-auto">
          <div className="flex-1 relative">
            <textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={streaming && !introLoaded}
              placeholder={
                streaming && !introLoaded
                  ? 'Loading patient briefing…'
                  : 'Ask anything about this patient… (Enter to send · Shift+Enter for newline)'
              }
              rows={1}
              className="w-full rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 pr-12
                         text-sm text-slate-800 placeholder:text-slate-400 resize-none
                         focus:outline-none focus:ring-2 focus:ring-indigo-500/30 focus:border-indigo-400
                         focus:bg-white transition-all duration-200 max-h-36 overflow-y-auto"
              style={{ minHeight: 44 }}
              onInput={(e) => {
                const t = e.currentTarget
                t.style.height = 'auto'
                t.style.height = `${Math.min(t.scrollHeight, 144)}px`
              }}
            />
          </div>

          {streaming ? (
            <button
              onClick={stopStreaming}
              className="flex-shrink-0 w-11 h-11 rounded-xl bg-rose-50 text-rose-500 border border-rose-100
                         flex items-center justify-center hover:bg-rose-100 transition-all duration-150"
              title="Stop generation"
            >
              <Square size={16} />
            </button>
          ) : (
            <button
              onClick={sendMessage}
              disabled={!input.trim() || !introLoaded}
              className="flex-shrink-0 w-11 h-11 rounded-xl gradient-brand text-white
                         flex items-center justify-center shadow-sm shadow-indigo-200
                         hover:opacity-90 active:opacity-80 transition-all duration-150
                         disabled:opacity-40 disabled:cursor-not-allowed disabled:shadow-none"
              title="Send message"
            >
              <Send size={16} />
            </button>
          )}
        </div>
        <p className="text-[11px] text-slate-400 mt-2 text-center">
          Powered by Ollama · Responses may take 20–60 s depending on model size
        </p>
      </div>
    </div>
  )
}
