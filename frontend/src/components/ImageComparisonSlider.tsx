import { useState, useRef } from 'react'
import { cls } from '../lib/utils'

interface ImageComparisonSliderProps {
  beforeImage: string
  afterImage: string
}

function ImageComparisonSlider({ beforeImage, afterImage }: ImageComparisonSliderProps) {
  const [sliderPosition, setSliderPosition] = useState(50)
  const [isDragging, setIsDragging] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)
  
  const getCurrentLabel = () => {
    return sliderPosition < 50 ? 'BEFORE' : 'AFTER'
  }

  const handleMove = (clientX: number) => {
    if (!containerRef.current) return
    
    const rect = containerRef.current.getBoundingClientRect()
    const x = clientX - rect.left
    const percentage = Math.max(0, Math.min(100, (x / rect.width) * 100))
    setSliderPosition(percentage)
  }

  const handleMouseDown = () => setIsDragging(true)
  const handleMouseUp = () => setIsDragging(false)
  
  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging) handleMove(e.clientX)
  }

  const handleTouchMove = (e: React.TouchEvent) => {
    if (e.touches.length > 0) handleMove(e.touches[0].clientX)
  }

  const currentLabel = getCurrentLabel()
  
  return (
    <div
      ref={containerRef}
      className="relative max-w-2xl overflow-hidden rounded-xl border border-zinc-200 shadow-lg dark:border-zinc-800 select-none cursor-ew-resize"
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
      onTouchEnd={handleMouseUp}
      onTouchMove={handleTouchMove}
    >
      {/* Base Layer - After Image */}
      <div className="relative w-full">
        <img
          src={afterImage}
          alt="After disaster"
          className="w-full h-auto block"
          draggable={false}
        />
      </div>

      {/* Top Layer - Before Image */}
      <div
        className="absolute inset-0 overflow-hidden"
        style={{ clipPath: `inset(0 ${100 - sliderPosition}% 0 0)` }}
      >
        <img
          src={beforeImage}
          alt="Before disaster"
          className="w-full h-full object-cover"
          draggable={false}
        />
      </div>

      {/* Dynamic Label */}
      <div className="absolute top-2 left-1/2 -translate-x-1/2 rounded-full bg-zinc-900/90 px-4 py-1.5 text-xs font-bold text-white backdrop-blur-sm dark:bg-white dark:text-zinc-900">
        {currentLabel}
      </div>

      {/* Slider Handle */}
      <div
        className="absolute inset-y-0 z-10 flex items-center"
        style={{ left: `${sliderPosition}%`, transform: 'translateX(-50%)' }}
        onMouseDown={handleMouseDown}
        onTouchStart={() => setIsDragging(true)}
      >
        <div className="h-full w-0.5 bg-white shadow-lg" />
        <div className="absolute flex h-10 w-10 -translate-x-1/2 items-center justify-center rounded-full border-2 border-white bg-zinc-900 shadow-xl dark:bg-white">
          <div className="flex gap-0.5">
            <div className="h-4 w-0.5 bg-white dark:bg-zinc-900" />
            <div className="h-4 w-0.5 bg-white dark:bg-zinc-900" />
          </div>
        </div>
      </div>
    </div>
  )
}

export default ImageComparisonSlider
