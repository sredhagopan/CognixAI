import type { ReactNode, MouseEventHandler } from 'react'
import { clsx } from '../../utils'

interface CardProps {
  children: ReactNode
  className?: string
  hoverable?: boolean
  padding?: 'sm' | 'md' | 'lg' | 'none'
  variant?: 'default' | 'glass' | 'elevated' | 'outline'
  onClick?: MouseEventHandler<HTMLDivElement>
}

const paddings = {
  none: '',
  sm:   'p-4',
  md:   'p-5',
  lg:   'p-6',
}

export default function Card({
  children,
  className,
  hoverable,
  padding = 'md',
  variant = 'default',
  onClick,
}: CardProps) {
  return (
    <div
      onClick={onClick}
      className={clsx(
        'rounded-2xl border transition-all duration-200',
        variant === 'default'  && 'bg-white border-slate-200 card-shadow',
        variant === 'glass'    && 'glass border-white/50 card-shadow',
        variant === 'elevated' && 'bg-white border-slate-200 shadow-md',
        variant === 'outline'  && 'bg-transparent border-slate-200',
        hoverable && 'hover:card-shadow-hover hover:-translate-y-0.5 cursor-pointer',
        paddings[padding],
        className,
      )}
    >
      {children}
    </div>
  )
}

export function CardHeader({
  children,
  className,
}: {
  children: ReactNode
  className?: string
}) {
  return (
    <div className={clsx('flex items-center justify-between mb-4', className)}>
      {children}
    </div>
  )
}

export function CardTitle({
  children,
  className,
  icon,
}: {
  children: ReactNode
  className?: string
  icon?: ReactNode
}) {
  return (
    <div className={clsx('flex items-center gap-2.5', className)}>
      {icon && (
        <div className="flex-shrink-0 w-7 h-7 rounded-lg bg-indigo-50 flex items-center justify-center text-indigo-500">
          {icon}
        </div>
      )}
      <h3 className="text-sm font-semibold text-slate-800 tracking-tight">{children}</h3>
    </div>
  )
}

export function CardDivider({ className }: { className?: string }) {
  return <div className={clsx('border-t border-slate-100 my-4', className)} />
}
