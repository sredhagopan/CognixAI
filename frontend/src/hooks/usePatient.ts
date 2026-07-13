import { useState, useCallback } from 'react'
import { fetchPatient } from '../services/api'
import type { PatientData } from '../types'

interface PatientState {
  data: PatientData | null
  loading: boolean
  error: string | null
}

export function usePatient() {
  const [state, setState] = useState<PatientState>({
    data: null, loading: false, error: null,
  })

  const loadPatient = useCallback(async (id: string) => {
    setState({ data: null, loading: true, error: null })
    try {
      const data = await fetchPatient(id.trim())
      setState({ data, loading: false, error: null })
      return data
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Failed to load patient'
      setState({ data: null, loading: false, error: msg })
      return null
    }
  }, [])

  const clearPatient = useCallback(() => {
    setState({ data: null, loading: false, error: null })
  }, [])

  return { ...state, loadPatient, clearPatient }
}
