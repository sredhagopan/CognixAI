import type { ReactNode } from 'react'
import { clsx } from '../../utils'

type BadgeVariant =
  | 'improving'
  | 'stable'
  | 'deteriorating'
  | 'default'
  | 'indigo'
  | 'purple'
  | 'sky'
  | 'low'
  | 'moderate'
  | 'high'
  | 'outline'

interface BadgeProps {
  children: ReactNode
  variant?: BadgeVariant
  className?: string
  dot?: boolean
}

const variants: Record<BadgeVariant, string> = {
  improving:     'bg-emerald-50 text-emerald-700 border border-emerald-200',
  stable:        'bg-amber-50 text-amber-700 border border-amber-200',
  deteriorating: 'bg-rose-50 text-rose-700 border border-rose-200',
  default:       'bg-slate-100 text-slate-600 border border-slate-200',
  indigo:        'bg-indigo-50 text-indigo-700 border border-indigo-200',
  purple:        'bg-violet-50 text-violet-700 border border-violet-200',
  sky:           'bg-sky-50 text-sky-700 border border-sky-200',
  low:           'bg-rose-50 text-rose-700 border border-rose-200',
  moderate:      'bg-amber-50 text-amber-700 border border-amber-200',
  high:          'bg-emerald-50 text-emerald-700 border border-emerald-200',
  outline:       'bg-transparent text-slate-600 border border-slate-300',
}

const dotColors: Record<BadgeVariant, string> = {
  improving:     'bg-emerald-500',
  stable:        'bg-amber-500',
  deteriorating: 'bg-rose-500',
  default:       'bg-slate-400',
  indigo:        'bg-indigo-500',
  purple:        'bg-violet-500',
  sky:           'bg-sky-500',
  low:           'bg-rose-500',
  moderate:      'bg-amber-500',
  high:          'bg-emerald-500',
  outline:       'bg-slate-400',
}

export default function Badge({
  children,
  variant = 'default',
  className,
  dot,
}: BadgeProps) {
  return (
    <span
      className={clsx(
        'inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-medium',
        variants[variant],
        className,
      )}
    >
      {dot && (
        <span className={clsx('w-1.5 h-1.5 rounded-full flex-shrink-0', dotColors[variant])} />
      )}
      {children}
    </span>
  )
}
