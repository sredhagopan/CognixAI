import { clsx } from '../../utils'

interface SpinnerProps {
  size?: 'xs' | 'sm' | 'md' | 'lg'
  className?: string
}

const sizes = {
  xs: 'w-3 h-3',
  sm: 'w-4 h-4',
  md: 'w-5 h-5',
  lg: 'w-8 h-8',
}

export default function Spinner({ size = 'md', className }: SpinnerProps) {
  return (
    <svg
      className={clsx('animate-spin text-indigo-600', sizes[size], className)}
      xmlns="http://www.w3.org/2000/svg"
      fill="none"
      viewBox="0 0 24 24"
    >
      <circle className="opacity-20" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="3" />
      <path className="opacity-80" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
    </svg>
  )
}

export function SkeletonLine({ className }: { className?: string }) {
  return (
    <div className={clsx('rounded-lg animate-shimmer', className)} />
  )
}

export function SkeletonCard({
  lines = 3,
  className,
}: {
  lines?: number
  className?: string
}) {
  return (
    <div className={clsx('bg-white rounded-2xl border border-slate-200 p-5 card-shadow', className)}>
      <div className="flex items-center gap-3 mb-4">
        <SkeletonLine className="w-7 h-7 rounded-lg flex-shrink-0" />
        <SkeletonLine className="h-4 w-32" />
      </div>
      <SkeletonLine className="h-8 w-16 mb-3" />
      {Array.from({ length: lines }).map((_, i) => (
        <SkeletonLine
          key={i}
          className={clsx('h-3 mb-2', i === lines - 1 ? 'w-3/5' : 'w-full')}
        />
      ))}
    </div>
  )
}

export function SkeletonText({ lines = 2, className }: { lines?: number; className?: string }) {
  return (
    <div className={clsx('space-y-2', className)}>
      {Array.from({ length: lines }).map((_, i) => (
        <SkeletonLine
          key={i}
          className={clsx('h-3', i === lines - 1 ? 'w-4/5' : 'w-full')}
        />
      ))}
    </div>
  )
}
