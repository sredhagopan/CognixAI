import { type ReactNode, useRef } from 'react'
import { clsx } from '../../utils'

export interface TabDef {
  id: string
  label: string
  icon?: ReactNode
}

interface TabsProps {
  tabs: TabDef[]
  active: string
  onChange: (id: string) => void
  className?: string
}

export default function Tabs({ tabs, active, onChange, className }: TabsProps) {
  const listRef = useRef<HTMLDivElement>(null)

  return (
    <div
      ref={listRef}
      className={clsx('flex overflow-x-auto gap-0.5', className)}
      style={{ scrollbarWidth: 'none' }}
    >
      {tabs.map((tab) => {
        const isActive = active === tab.id
        return (
          <button
            key={tab.id}
            onClick={() => onChange(tab.id)}
            className={clsx(
              'relative flex items-center gap-2 px-4 py-3 text-sm whitespace-nowrap',
              'transition-all duration-150 select-none outline-none',
              'border-b-2',
              isActive
                ? 'border-indigo-600 text-indigo-600 font-semibold'
                : 'border-transparent text-slate-500 hover:text-slate-800 hover:border-slate-300 font-normal',
            )}
          >
            {tab.icon && (
              <span className={clsx('transition-colors', isActive ? 'text-indigo-500' : 'text-slate-400')}>
                {tab.icon}
              </span>
            )}
            {tab.label}
          </button>
        )
      })}
    </div>
  )
}
