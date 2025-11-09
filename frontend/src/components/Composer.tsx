import { useState, useRef, useEffect, forwardRef, useImperativeHandle } from 'react'
import { Send, Loader2, Plus, ImagePlus, Map } from 'lucide-react'
import { cls } from '../lib/utils'
import axios from 'axios'
import MapPicker from './MapPicker'

interface ComposerProps {
  onSend?: (text: string) => Promise<void>
  onImageUpload?: (file: File) => Promise<void>
  busy?: boolean
}

const Composer = forwardRef<any, ComposerProps>(function Composer({ onSend, onImageUpload, busy }, ref) {
  const [value, setValue] = useState('')
  const [sending, setSending] = useState(false)
  const [lineCount, setLineCount] = useState(1)
  const [beforeImage, setBeforeImage] = useState<File | null>(null)
  const [afterImage, setAfterImage] = useState<File | null>(null)
  const [beforePreview, setBeforePreview] = useState<string | null>(null)
  const [afterPreview, setAfterPreview] = useState<string | null>(null)
  const [uploadingFor, setUploadingFor] = useState<'before' | 'after' | null>(null)
  const [showMapPicker, setShowMapPicker] = useState(false)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (inputRef.current) {
      const textarea = inputRef.current
      const lineHeight = 20
      const minHeight = 40

      textarea.style.height = 'auto'
      const scrollHeight = textarea.scrollHeight
      const calculatedLines = Math.max(1, Math.floor((scrollHeight - 16) / lineHeight))

      setLineCount(calculatedLines)

      if (calculatedLines <= 12) {
        textarea.style.height = `${Math.max(minHeight, scrollHeight)}px`
        textarea.style.overflowY = 'hidden'
      } else {
        textarea.style.height = `${minHeight + 11 * lineHeight}px`
        textarea.style.overflowY = 'auto'
      }
    }
  }, [value])

  useImperativeHandle(
    ref,
    () => ({
      insertTemplate: (templateContent: string) => {
        setValue((prev) => {
          const newValue = prev ? `${prev}\n\n${templateContent}` : templateContent
          setTimeout(() => {
            inputRef.current?.focus()
            const length = newValue.length
            inputRef.current?.setSelectionRange(length, length)
          }, 0)
          return newValue
        })
      },
      focus: () => {
        inputRef.current?.focus()
      },
    }),
    [],
  )

  async function handleSend() {
    if ((!value.trim() && !beforeImage && !afterImage) || sending) return
    if (beforeImage && afterImage && !onImageUpload) return
    
    // Need both images if uploading
    if ((beforeImage || afterImage) && !(beforeImage && afterImage)) {
      alert('Please upload both before and after images')
      return
    }
    
    setSending(true)
    try {
      // If there are images, send them
      if (beforeImage && afterImage && onImageUpload) {
        const formData = { before: beforeImage, after: afterImage }
        await onImageUpload(formData as any)
        setBeforeImage(null)
        setAfterImage(null)
        setBeforePreview(null)
        setAfterPreview(null)
      } else if (value.trim()) {
        // Otherwise just send text
        await onSend?.(value)
      }
      setValue('')
      inputRef.current?.focus()
    } finally {
      setSending(false)
    }
  }

  const handleImageSelect = (file: File) => {
    if (uploadingFor === 'before') {
      if (beforePreview) URL.revokeObjectURL(beforePreview)
      setBeforeImage(file)
      setBeforePreview(URL.createObjectURL(file))
    } else if (uploadingFor === 'after') {
      if (afterPreview) URL.revokeObjectURL(afterPreview)
      setAfterImage(file)
      setAfterPreview(URL.createObjectURL(file))
    }
    setUploadingFor(null)
  }

  const handleRemoveImage = (type: 'before' | 'after') => {
    if (type === 'before') {
      if (beforePreview) URL.revokeObjectURL(beforePreview)
      setBeforeImage(null)
      setBeforePreview(null)
    } else {
      if (afterPreview) URL.revokeObjectURL(afterPreview)
      setAfterImage(null)
      setAfterPreview(null)
    }
  }

  return (
    <div className="border-t border-zinc-200/60 p-4 dark:border-zinc-800">
      <div
        className={cls(
          'mx-auto flex flex-col rounded-2xl border bg-white shadow-sm dark:bg-zinc-950 transition-all duration-200',
          'max-w-3xl border-zinc-300 dark:border-zinc-700 p-3',
        )}
      >
        {(beforePreview || afterPreview) && (
          <div className="mb-3 flex gap-3 animate-fadeIn">
            {beforePreview && (
              <div className="relative">
                <div className="text-[10px] text-zinc-500 mb-1 font-medium">BEFORE</div>
                <img
                  src={beforePreview}
                  alt="Before"
                  className="h-20 w-20 rounded-lg border border-zinc-200 object-cover dark:border-zinc-700"
                />
                <button
                  onClick={() => handleRemoveImage('before')}
                  className="absolute -right-2 -top-2 flex h-5 w-5 items-center justify-center rounded-full bg-zinc-900 text-white hover:bg-zinc-700 dark:bg-white dark:text-zinc-900 dark:hover:bg-zinc-200"
                >
                  ×
                </button>
              </div>
            )}
            {afterPreview && (
              <div className="relative">
                <div className="text-[10px] text-zinc-500 mb-1 font-medium">AFTER</div>
                <img
                  src={afterPreview}
                  alt="After"
                  className="h-20 w-20 rounded-lg border border-zinc-200 object-cover dark:border-zinc-700"
                />
                <button
                  onClick={() => handleRemoveImage('after')}
                  className="absolute -right-2 -top-2 flex h-5 w-5 items-center justify-center rounded-full bg-zinc-900 text-white hover:bg-zinc-700 dark:bg-white dark:text-zinc-900 dark:hover:bg-zinc-200"
                >
                  ×
                </button>
              </div>
            )}
          </div>
        )}
        
        <div className="flex-1 relative">
          <textarea
            ref={inputRef}
            value={value}
            onChange={(e) => setValue(e.target.value)}
            placeholder={(beforeImage || afterImage) ? "Add a message about these images..." : "How can I help you today?"}
            rows={1}
            className={cls(
              'w-full resize-none bg-transparent text-sm outline-none placeholder:text-zinc-400 transition-all duration-200',
              'px-0 py-2 min-h-[40px] text-left',
            )}
            style={{
              height: 'auto',
              overflowY: lineCount > 12 ? 'auto' : 'hidden',
            }}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                handleSend()
              }
            }}
          />
        </div>

        <div className="flex items-center justify-between mt-2">
          <div className="flex gap-1">
            <button
              onClick={() => setShowMapPicker(true)}
              className="inline-flex shrink-0 items-center gap-1 justify-center rounded-full px-2 py-1.5 text-xs text-zinc-500 hover:bg-zinc-100 hover:text-zinc-700 dark:text-zinc-400 dark:hover:bg-zinc-800 dark:hover:text-zinc-300 transition-colors"
              title="Pick coordinates from map"
            >
              <Map className="h-3.5 w-3.5" />
              <span className="text-[10px] font-medium">Map</span>
            </button>
            <button
              onClick={() => { setUploadingFor('before'); fileInputRef.current?.click(); }}
              disabled={!!beforeImage}
              className="inline-flex shrink-0 items-center gap-1 justify-center rounded-full px-2 py-1.5 text-xs text-zinc-500 hover:bg-zinc-100 hover:text-zinc-700 dark:text-zinc-400 dark:hover:bg-zinc-800 dark:hover:text-zinc-300 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              title="Upload before image"
            >
              <ImagePlus className="h-3.5 w-3.5" />
              <span className="text-[10px] font-medium">Before</span>
            </button>
            <button
              onClick={() => { setUploadingFor('after'); fileInputRef.current?.click(); }}
              disabled={!!afterImage}
              className="inline-flex shrink-0 items-center gap-1 justify-center rounded-full px-2 py-1.5 text-xs text-zinc-500 hover:bg-zinc-100 hover:text-zinc-700 dark:text-zinc-400 dark:hover:bg-zinc-800 dark:hover:text-zinc-300 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              title="Upload after image"
            >
              <ImagePlus className="h-3.5 w-3.5" />
              <span className="text-[10px] font-medium">After</span>
            </button>
          </div>

          <div className="flex items-center gap-1 shrink-0">
            <button
              onClick={handleSend}
              disabled={sending || busy || (!value.trim() && !(beforeImage && afterImage)) || ((beforeImage || afterImage) && !(beforeImage && afterImage))}
              className={cls(
                'inline-flex shrink-0 items-center gap-2 rounded-full bg-zinc-900 px-3 py-2 text-sm font-medium text-white shadow-sm transition hover:bg-zinc-800 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 dark:bg-white dark:text-zinc-900',
                (sending || busy || (!value.trim() && !(beforeImage && afterImage))) && 'opacity-50 cursor-not-allowed',
              )}
            >
              {sending || busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
            </button>
          </div>
        </div>
      </div>

      <div className="mx-auto mt-2 max-w-3xl px-1 text-[11px] text-zinc-500 dark:text-zinc-400">
        Press{' '}
        <kbd className="rounded border border-zinc-300 bg-zinc-50 px-1 dark:border-zinc-600 dark:bg-zinc-800">
          Enter
        </kbd>{' '}
        to send ·{' '}
        <kbd className="rounded border border-zinc-300 bg-zinc-50 px-1 dark:border-zinc-600 dark:bg-zinc-800">
          Shift
        </kbd>
        +
        <kbd className="rounded border border-zinc-300 bg-zinc-50 px-1 dark:border-zinc-600 dark:bg-zinc-800">
          Enter
        </kbd>{' '}
        for newline
      </div>

      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        style={{ display: 'none' }}
        onChange={(e) => {
          const file = e.target.files?.[0]
          if (file) {
            handleImageSelect(file)
            e.target.value = '' // Reset input
          }
        }}
      />
      
      {showMapPicker && (
        <MapPicker
          onClose={() => setShowMapPicker(false)}
          onSelect={(lat, lon) => {
            const coordText = `at coordinates ${lat}°N, ${lon}°W`
            setValue((prev) => prev ? `${prev} ${coordText}` : coordText)
            inputRef.current?.focus()
          }}
        />
      )}
    </div>
  )
})

export default Composer
