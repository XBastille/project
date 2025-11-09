import { useState, useRef, useEffect, forwardRef } from 'react'
import axios from 'axios'
import { cls, timeAgo } from '../lib/utils'
import Message from './Message'
import Composer from './Composer'
import ImageComparisonSlider from './ImageComparisonSlider'
import { MapPin, Image as ImageIcon } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  createdAt: string
  beforeImage?: string
  afterImage?: string
  segmentationImage?: string
  assessment?: AssessmentResult
}

interface AssessmentResult {
  location: string
  damage_level: string
  confidence: string
  decision: string
  priority: string
  damage_probabilities: {
    no_building: string
    no_damage: string
    minor_damage: string
    major_damage: string
    destroyed: string
  }
}

function ThinkingMessage({ onPause }: { onPause?: () => void }) {
  return (
    <Message role="assistant">
      <div className="flex items-center gap-3">
        <div className="flex items-center gap-1">
          <div className="h-2 w-2 animate-bounce rounded-full bg-zinc-400 [animation-delay:-0.3s]"></div>
          <div className="h-2 w-2 animate-bounce rounded-full bg-zinc-400 [animation-delay:-0.15s]"></div>
          <div className="h-2 w-2 animate-bounce rounded-full bg-zinc-400"></div>
        </div>
        <span className="text-sm text-zinc-500">Analyzing...</span>
      </div>
    </Message>
  )
}

function ChatInterface() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [isThinking, setIsThinking] = useState(false)
  const composerRef = useRef<any>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const handleSend = async (text: string) => {
    if (!text.trim()) return

    const userMessage: ChatMessage = {
      id: Math.random().toString(36).slice(2),
      role: 'user',
      content: text,
      createdAt: new Date().toISOString(),
    }

    setMessages((prev) => [...prev, userMessage])
    setIsThinking(true)

    try {
      const response = await axios.post('/api/chat', { message: text })

      const assistantMessage: ChatMessage = {
        id: Math.random().toString(36).slice(2),
        role: 'assistant',
        content: response.data.response,
        createdAt: new Date().toISOString(),
        assessment: response.data.assessment,
        beforeImage: response.data.beforeImageUrl,
        afterImage: response.data.afterImageUrl,
        segmentationImage: response.data.segmentationImageUrl,
      }

      setMessages((prev) => [...prev, assistantMessage])
    } catch (error) {
      console.error('Chat error:', error)
      const errorMessage: ChatMessage = {
        id: Math.random().toString(36).slice(2),
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.',
        createdAt: new Date().toISOString(),
      }
      setMessages((prev) => [...prev, errorMessage])
    } finally {
      setIsThinking(false)
    }
  }

  const handleExampleClick = (example: string) => {
    composerRef.current?.insertTemplate(example)
  }

  const handleImageUpload = async (formData: any) => {
    const beforeUrl = URL.createObjectURL(formData.before)
    const afterUrl = URL.createObjectURL(formData.after)
    
    const userMessage: ChatMessage = {
      id: Math.random().toString(36).slice(2),
      role: 'user',
      content: 'Analyze disaster damage between these images',
      createdAt: new Date().toISOString(),
      beforeImage: beforeUrl,
      afterImage: afterUrl,
    }

    setMessages((prev) => [...prev, userMessage])
    setIsThinking(true)

    try {
      const uploadData = new FormData()
      uploadData.append('before', formData.before)
      uploadData.append('after', formData.after)

      const response = await axios.post('/api/analyze-image', uploadData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      })

      const assistantMessage: ChatMessage = {
        id: Math.random().toString(36).slice(2),
        role: 'assistant',
        content: response.data.response,
        createdAt: new Date().toISOString(),
        assessment: response.data.assessment,
        beforeImage: response.data.beforeImageUrl,
        afterImage: response.data.afterImageUrl,
        segmentationImage: response.data.segmentationImageUrl,
      }

      setMessages((prev) => [...prev, assistantMessage])
    } catch (error) {
      console.error('Image analysis error:', error)
      const errorMessage: ChatMessage = {
        id: Math.random().toString(36).slice(2),
        role: 'assistant',
        content: 'Failed to analyze the images. Please try again.',
        createdAt: new Date().toISOString(),
      }
      setMessages((prev) => [...prev, errorMessage])
    } finally {
      setIsThinking(false)
    }
  }

  const handleSampleAnalysis = async () => {
    const userMessage: ChatMessage = {
      id: Math.random().toString(36).slice(2),
      role: 'user',
      content: 'Run sample disaster analysis for Hurricane Ian',
      createdAt: new Date().toISOString(),
    }

    setMessages((prev) => [...prev, userMessage])
    setIsThinking(true)

    try {
      const response = await axios.post('/api/sample-analysis')

      const assistantMessage: ChatMessage = {
        id: Math.random().toString(36).slice(2),
        role: 'assistant',
        content: response.data.response,
        createdAt: new Date().toISOString(),
        assessment: response.data.assessment,
        beforeImage: response.data.beforeImageUrl,
        afterImage: response.data.afterImageUrl,
        segmentationImage: response.data.segmentationImageUrl,
      }

      setMessages((prev) => [...prev, assistantMessage])
    } catch (error) {
      console.error('Sample analysis error:', error)
    } finally {
      setIsThinking(false)
    }
  }

  const isEmpty = messages.length === 0

  return (
    <div className="flex h-screen w-full flex-col bg-white text-zinc-900 dark:bg-zinc-950 dark:text-zinc-100">
      {isEmpty ? (
        <div className="flex flex-1 flex-col items-center justify-center p-6">
          <div className="mb-8 text-center">
            <div className="mb-6">
              <img 
                src="/logo.png" 
                alt="OrbitalClaim Logo" 
                className="mx-auto h-32 w-32 object-contain"
              />
            </div>
            <h1 className="mb-2 text-4xl font-bold tracking-tight bg-gradient-to-r from-blue-600 to-cyan-600 bg-clip-text text-transparent">
              OrbitalClaim
            </h1>
            <p className="text-sm text-zinc-500 dark:text-zinc-400">
              Satellite-powered disaster damage assessment
            </p>
          </div>

          <div className="mb-8 grid w-full max-w-2xl gap-3 sm:grid-cols-2">
            <button
              onClick={() => handleExampleClick('Assess damage from Hurricane Ian in Fort Myers, Florida')}
              className="group flex flex-col gap-2 rounded-2xl border border-zinc-200 bg-white p-4 text-left transition-all hover:border-zinc-300 hover:shadow-sm dark:border-zinc-800 dark:bg-zinc-900 dark:hover:border-zinc-700"
            >
              <div className="flex items-center gap-2">
                <MapPin className="h-4 w-4 text-blue-500" />
                <span className="text-sm font-medium">Location Assessment</span>
              </div>
              <p className="text-xs text-zinc-500 dark:text-zinc-400">
                Analyze disaster damage by location and dates
              </p>
            </button>

            <button
              onClick={handleSampleAnalysis}
              className="group flex flex-col gap-2 rounded-2xl border border-zinc-200 bg-white p-4 text-left transition-all hover:border-zinc-300 hover:shadow-sm dark:border-zinc-800 dark:bg-zinc-900 dark:hover:border-zinc-700"
            >
              <div className="flex items-center gap-2">
                <ImageIcon className="h-4 w-4 text-zinc-500" />
                <span className="text-sm font-medium">Sample Analysis</span>
              </div>
              <p className="text-xs text-zinc-500 dark:text-zinc-400">
                Run damage detection on sample disaster imagery
              </p>
            </button>
          </div>
        </div>
      ) : (
        <div className="flex min-h-0 flex-1 flex-col">
          <div className="mx-auto w-full max-w-3xl flex-1 space-y-5 overflow-y-auto px-4 py-6 sm:px-8 scrollbar-hide">
            {messages.map((msg) => (
              <div key={msg.id} className="space-y-4">
                <Message role={msg.role}>
                  <div className="prose prose-sm prose-neutral dark:prose-invert max-w-none prose-headings:text-zinc-900 dark:prose-headings:text-zinc-100 prose-p:text-zinc-900 dark:prose-p:text-zinc-100 prose-strong:text-zinc-900 dark:prose-strong:text-zinc-100 prose-li:text-zinc-900 dark:prose-li:text-zinc-100">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>{msg.content}</ReactMarkdown>
                  </div>
                </Message>
                {msg.assessment && <AssessmentCard assessment={msg.assessment} />}
                
                {/* Damage Detection Image (No Slider) */}
                {msg.segmentationImage && (
                  <div className="ml-10 mt-3 animate-fadeIn">
                    <div className="relative max-w-2xl overflow-hidden rounded-xl border border-zinc-200 shadow-lg dark:border-zinc-800">
                      <img
                        src={msg.segmentationImage}
                        alt="Damage Detection"
                        className="w-full h-auto"
                      />
                      <div className="absolute top-2 left-1/2 -translate-x-1/2 rounded-full bg-orange-500/90 px-4 py-1.5 text-xs font-bold text-white backdrop-blur-sm">
                        DAMAGE DETECTION
                      </div>
                    </div>
                    <div className="mt-2 text-center text-xs text-zinc-500 dark:text-zinc-400">
                      AI-detected building damage visualization
                    </div>
                  </div>
                )}
                
                {/* Before/After Comparison Slider */}
                {msg.beforeImage && msg.afterImage && (
                  <div className="ml-10 mt-3 animate-fadeIn">
                    <ImageComparisonSlider 
                      beforeImage={msg.beforeImage} 
                      afterImage={msg.afterImage}
                    />
                    <div className="mt-2 text-center text-xs text-zinc-500 dark:text-zinc-400">
                      Drag slider to compare: <span className="text-blue-500 font-semibold">Before</span> ↔ <span className="text-red-500 font-semibold">After</span>
                    </div>
                  </div>
                )}
              </div>
            ))}
            {isThinking && <ThinkingMessage />}
            <div ref={messagesEndRef} />
          </div>
        </div>
      )}

      <Composer ref={composerRef} onSend={handleSend} onImageUpload={handleImageUpload} busy={isThinking} />
    </div>
  )
}

function AssessmentCard({ assessment }: { assessment: AssessmentResult }) {
  const getDecisionColor = (decision: string) => {
    if (decision === 'APPROVE') return '#10b981'
    if (decision === 'REJECT') return '#ef4444'
    return '#f59e0b'
  }

  const getPriorityColor = (priority: string) => {
    if (priority === 'HIGH') return '#ef4444'
    if (priority === 'MEDIUM') return '#f59e0b'
    return '#6b7280'
  }

  return (
    <div className="ml-10 rounded-2xl border border-zinc-200 bg-white p-4 shadow-sm dark:border-zinc-800 dark:bg-zinc-900">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <h3 className="text-sm font-semibold">Assessment Results</h3>
        <div className="flex gap-2">
          <span
            className="rounded-full px-2 py-1 text-[10px] font-bold uppercase tracking-wide text-white"
            style={{ backgroundColor: getDecisionColor(assessment.decision) }}
          >
            {assessment.decision}
          </span>
          <span
            className="rounded-full px-2 py-1 text-[10px] font-bold uppercase tracking-wide text-white"
            style={{ backgroundColor: getPriorityColor(assessment.priority) }}
          >
            {assessment.priority}
          </span>
        </div>
      </div>

      <div className="grid gap-3 sm:grid-cols-3">
        <div>
          <div className="text-[10px] uppercase tracking-wide text-zinc-500">Location</div>
          <div className="text-sm font-medium">{assessment.location}</div>
        </div>
        <div>
          <div className="text-[10px] uppercase tracking-wide text-zinc-500">Damage Level</div>
          <div className="text-sm font-medium">{assessment.damage_level}</div>
        </div>
        <div>
          <div className="text-[10px] uppercase tracking-wide text-zinc-500">Confidence</div>
          <div className="text-sm font-medium">{assessment.confidence}</div>
        </div>
      </div>

      <div className="mt-3 border-t border-zinc-200 pt-3 dark:border-zinc-800">
        <div className="mb-2 text-[10px] uppercase tracking-wide text-zinc-500">Classification:</div>
        {Object.entries(assessment.damage_probabilities).map(([key, value]) => (
          <div key={key} className="mb-1.5 flex items-center gap-2 text-xs">
            <span className="w-24 capitalize text-zinc-600 dark:text-zinc-400">
              {key.replace(/_/g, ' ')}
            </span>
            <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-zinc-100 dark:bg-zinc-800">
              <div
                className="h-full rounded-full bg-gradient-to-r from-blue-500 to-purple-500"
                style={{ width: value }}
              ></div>
            </div>
            <span className="w-12 text-right text-zinc-500">{value}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

export default ChatInterface

