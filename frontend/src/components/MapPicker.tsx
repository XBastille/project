import { useState, useEffect } from 'react'
import { MapContainer, TileLayer, Marker, useMapEvents } from 'react-leaflet'
import { X, MapPin, Copy, Check } from 'lucide-react'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'

// Fix default marker icon issue with webpack
delete (L.Icon.Default.prototype as any)._getIconUrl
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
})

interface MapPickerProps {
  onClose: () => void
  onSelect: (lat: number, lon: number) => void
}

function LocationMarker({ position, setPosition }: { position: { lat: number; lon: number }; setPosition: (pos: { lat: number; lon: number }) => void }) {
  useMapEvents({
    click(e) {
      setPosition({
        lat: Number(e.latlng.lat.toFixed(4)),
        lon: Number(e.latlng.lng.toFixed(4))
      })
    },
  })

  return position ? <Marker position={[position.lat, position.lon]} /> : null
}

function MapPicker({ onClose, onSelect }: MapPickerProps) {
  const [position, setPosition] = useState({ lat: 26.6406, lon: -81.8723 })
  const [copied, setCopied] = useState(false)

  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose()
    }
    window.addEventListener('keydown', handleEscape)
    return () => window.removeEventListener('keydown', handleEscape)
  }, [onClose])

  const handleCopyCoordinates = () => {
    const coordText = `${position.lat}°N, ${position.lon}°W`
    navigator.clipboard.writeText(coordText)
    setCopied(true)
    setTimeout(() => setCopied(false), 2000)
  }

  const handleSelectAndClose = () => {
    onSelect(position.lat, position.lon)
    onClose()
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm">
      <div className="relative h-full w-full">
        <button
          onClick={onClose}
          className="absolute right-4 top-4 z-[1000] rounded-full bg-white p-2 shadow-lg hover:bg-zinc-100 dark:bg-zinc-900 dark:hover:bg-zinc-800"
        >
          <X className="h-5 w-5" />
        </button>

        <MapContainer
          center={[position.lat, position.lon]}
          zoom={13}
          style={{ height: '100%', width: '100%' }}
          zoomControl={false}
        >
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />
          <LocationMarker position={position} setPosition={setPosition} />
        </MapContainer>

        <div className="absolute left-1/2 top-4 -translate-x-1/2 z-[1000]">
          <div className="rounded-2xl bg-white px-6 py-4 shadow-2xl dark:bg-zinc-900">
            <div className="mb-3 flex items-center gap-2">
              <MapPin className="h-5 w-5 text-blue-600" />
              <span className="text-sm font-semibold">Select Coordinates</span>
            </div>
            
            <div className="mb-4 rounded-lg bg-zinc-100 px-4 py-3 font-mono text-sm dark:bg-zinc-800">
              <div className="text-zinc-600 dark:text-zinc-400">Latitude: <span className="font-bold text-zinc-900 dark:text-zinc-100">{position.lat}°</span></div>
              <div className="text-zinc-600 dark:text-zinc-400">Longitude: <span className="font-bold text-zinc-900 dark:text-zinc-100">{position.lon}°</span></div>
            </div>

            <div className="flex gap-2">
              <button
                onClick={handleCopyCoordinates}
                className="flex flex-1 items-center justify-center gap-2 rounded-lg bg-zinc-200 px-4 py-2 text-sm font-medium transition-colors hover:bg-zinc-300 dark:bg-zinc-700 dark:hover:bg-zinc-600"
              >
                {copied ? (
                  <>
                    <Check className="h-4 w-4 text-green-600" />
                    Copied!
                  </>
                ) : (
                  <>
                    <Copy className="h-4 w-4" />
                    Copy
                  </>
                )}
              </button>
              
              <button
                onClick={handleSelectAndClose}
                className="flex flex-1 items-center justify-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-blue-700"
              >
                <MapPin className="h-4 w-4" />
                Use Location
              </button>
            </div>
          </div>
        </div>

        <div className="absolute bottom-4 left-1/2 -translate-x-1/2 rounded-full bg-black/60 px-4 py-2 text-xs text-white backdrop-blur-sm z-[1000]">
          Click anywhere on the map to select coordinates • Press ESC to close
        </div>
      </div>
    </div>
  )
}

export default MapPicker
