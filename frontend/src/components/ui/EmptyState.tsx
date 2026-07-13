import type { ReactNode } from 'react'
import { clsx } from '../../utils'

interface EmptyStateProps {
  icon?: ReactNode
  title: string
  description?: string
  action?: ReactNode
  className?: string
  size?: 'sm' | 'md' | 'lg'
}

export default function EmptyState({
  icon,
  title,
  description,
  action,
  className,
  size = 'md',
}: EmptyStateProps) {
  return (
    <div
      className={clsx(
        'flex flex-col items-center justify-center text-center',
        size === 'sm' && 'py-8 px-4',
        size === 'md' && 'py-16 px-6',
        size === 'lg' && 'py-24 px-8',
        className,
      )}
    >
      {icon && (
        <div className="mb-4 flex items-center justify-center w-14 h-14 rounded-2xl bg-slate-100 text-slate-400">
          {icon}
        </div>
      )}
      <h3
        className={clsx(
          'font-semibold text-slate-800',
          size === 'sm' ? 'text-sm' : 'text-base',
        )}
      >
        {title}
      </h3>
      {description && (
        <p
          className={clsx(
            'mt-1.5 text-slate-500 max-w-sm leading-relaxed',
            size === 'sm' ? 'text-xs' : 'text-sm',
          )}
        >
          {description}
        </p>
      )}
      {action && <div className="mt-5">{action}</div>}
    </div>
  )
}
