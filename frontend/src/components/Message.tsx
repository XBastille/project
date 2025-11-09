import { ReactNode } from 'react'
import { cls } from '../lib/utils'

interface MessageProps {
  role: 'user' | 'assistant'
  children: ReactNode
}

export default function Message({ role, children }: MessageProps) {
  const isUser = role === 'user'
  return (
    <div className={cls('flex gap-3', isUser ? 'justify-end' : 'justify-start')}>
      {!isUser && (
        <div className="mt-0.5 h-7 w-7 flex-shrink-0 overflow-hidden rounded-full">
          <img 
            src="/logo.jpg" 
            alt="OrbitalClaim AI" 
            className="h-full w-full object-cover"
          />
        </div>
      )}
      <div
        className={cls(
          'max-w-[80%] rounded-2xl px-3 py-2 text-sm shadow-sm [&_*]:!text-inherit',
          isUser
            ? 'bg-zinc-900 !text-white dark:bg-white dark:!text-zinc-900'
            : 'bg-white !text-zinc-900 dark:bg-zinc-900 dark:!text-zinc-100 border border-zinc-200 dark:border-zinc-800',
        )}
      >
        {children}
      </div>
      {isUser && (
        <div className="mt-0.5 grid h-7 w-7 place-items-center rounded-full bg-zinc-900 text-[10px] font-bold text-white dark:bg-white dark:text-zinc-900">
          U
        </div>
      )}
    </div>
  )
}
