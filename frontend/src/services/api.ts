import type {
  PatientData,
  SimulationResult,
  FeatureImportanceRow,
  SimulationMeta,
} from '../types'

const BASE = '/api'

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`)
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }))
    throw new Error(err.error || res.statusText)
  }
  return res.json()
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify(body),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }))
    throw new Error(err.error || res.statusText)
  }
  return res.json()
}

// ── Patients ──────────────────────────────────────────────────────────────────

export const fetchPatients = (): Promise<string[]> =>
  get('/patients')

export const fetchPatient = (id: string): Promise<PatientData> =>
  get(`/patient/${encodeURIComponent(id)}`)

// ── Simulation ────────────────────────────────────────────────────────────────

export const runSimulation = (
  patientId: string,
  mode: 'specific' | 'best_single' | 'best_combination',
  changes?: Record<string, number>,
): Promise<SimulationResult> =>
  post(`/patient/${encodeURIComponent(patientId)}/simulate`, { mode, changes })

// ── Feature importance & meta ─────────────────────────────────────────────────

export const fetchFeatureImportance = (): Promise<FeatureImportanceRow[]> =>
  get('/feature-importance')

export const fetchSimulationMeta = (): Promise<SimulationMeta> =>
  get('/simulation-meta')

export const fetchPlots = (): Promise<string[]> =>
  get('/plots')

export const plotUrl = (filename: string): string =>
  `${BASE}/plots/${encodeURIComponent(filename)}`

// ── Health ────────────────────────────────────────────────────────────────────

export const fetchHealth = () =>
  get<{ status: string; ollama_model: string; patients_loaded: number }>('/health')

// ── SSE helpers ───────────────────────────────────────────────────────────────

export type SseEvent =
  | { type: 'token';      content: string }
  | { type: 'metadata';   clean_content: string; response: string }
  | { type: 'sim_result'; result: SimulationResult }
  | { type: 'sim_warning'; message: string }
  | { type: 'done' }
  | { type: 'error';      message: string }

/**
 * Opens an SSE stream for the intro generation.
 * Returns an abort function.
 */
export function streamIntro(
  patientId: string,
  onEvent: (e: SseEvent) => void,
): () => void {
  const ctrl = new AbortController()

  fetch(`${BASE}/patient/${encodeURIComponent(patientId)}/intro`, {
    signal: ctrl.signal,
  }).then(async (res) => {
    const reader = res.body!.getReader()
    const dec    = new TextDecoder()
    let buf      = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buf += dec.decode(value, { stream: true })
      const parts = buf.split('\n\n')
      buf = parts.pop() ?? ''
      for (const part of parts) {
        const line = part.replace(/^data: /, '').trim()
        if (!line) continue
        try { onEvent(JSON.parse(line)) } catch { /* ignore */ }
      }
    }
  }).catch((err) => {
    if (err.name !== 'AbortError') {
      onEvent({ type: 'error', message: String(err) })
    }
  })

  return () => ctrl.abort()
}

/**
 * Opens an SSE stream for a chat turn.
 * Returns an abort function.
 */
export function streamChat(
  patientId: string,
  message: string,
  history: { role: 'user' | 'assistant'; content: string }[],
  onEvent: (e: SseEvent) => void,
): () => void {
  const ctrl = new AbortController()

  fetch(`${BASE}/patient/${encodeURIComponent(patientId)}/chat`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ message, history }),
    signal:  ctrl.signal,
  }).then(async (res) => {
    const reader = res.body!.getReader()
    const dec    = new TextDecoder()
    let buf      = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) break
      buf += dec.decode(value, { stream: true })
      const parts = buf.split('\n\n')
      buf = parts.pop() ?? ''
      for (const part of parts) {
        const line = part.replace(/^data: /, '').trim()
        if (!line) continue
        try { onEvent(JSON.parse(line)) } catch { /* ignore */ }
      }
    }
  }).catch((err) => {
    if (err.name !== 'AbortError') {
      onEvent({ type: 'error', message: String(err) })
    }
  })

  return () => ctrl.abort()
}
