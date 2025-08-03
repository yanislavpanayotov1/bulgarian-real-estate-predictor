import { useEffect, useState } from 'react'
import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet'
import { useQuery } from 'react-query'
import axios from 'axios'
import L from 'leaflet'
import { Home, MapPin, DollarSign, Maximize } from 'lucide-react'

// Fix for default markers in react-leaflet
delete (L.Icon.Default.prototype as any)._getIconUrl
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
})

// Custom property icon
const createPropertyIcon = (price: number, avgPrice: number) => {
  const isExpensive = price > avgPrice * 1.2
  const isCheap = price < avgPrice * 0.8
  
  let color = '#3b82f6' // default blue
  if (isExpensive) color = '#ef4444' // red for expensive
  if (isCheap) color = '#10b981' // green for cheap
  
  return L.divIcon({
    html: `
      <div style="
        background-color: ${color};
        width: 24px;
        height: 24px;
        border-radius: 50%;
        border: 2px solid white;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        display: flex;
        align-items: center;
        justify-content: center;
      ">
        <div style="
          width: 8px;
          height: 8px;
          background-color: white;
          border-radius: 50%;
        "></div>
      </div>
    `,
    className: 'custom-property-marker',
    iconSize: [24, 24],
    iconAnchor: [12, 12],
    popupAnchor: [0, -12],
  })
}

interface Property {
  id?: string
  price?: number
  size?: number
  rooms?: number
  floor?: number
  year_built?: number
  city?: string
  neighborhood?: string
  latitude?: number
  longitude?: number
  features?: string[]
  url?: string
  description?: string
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

// Map center component to handle centering on new data
const MapCenter: React.FC<{ center: [number, number] }> = ({ center }) => {
  const map = useMap()
  
  useEffect(() => {
    map.setView(center, map.getZoom())
  }, [map, center])
  
  return null
}

const PropertyMap: React.FC = () => {
  const [mapCenter, setMapCenter] = useState<[number, number]>([42.7339, 25.4858]) // Bulgaria center
  const [selectedCity, setSelectedCity] = useState<string>('')

  // Fetch properties for the map
  const { data: properties = [], isLoading, error } = useQuery<Property[]>(
    ['properties', selectedCity],
    async () => {
      const params = new URLSearchParams()
      if (selectedCity) params.append('city', selectedCity)
      params.append('limit', '100')
      
      console.log('Fetching properties from:', `${API_BASE_URL}/properties?${params.toString()}`)
      const response = await axios.get(`${API_BASE_URL}/properties?${params.toString()}`)
      console.log('Properties response:', response.data)
      
      // Validate response
      if (!Array.isArray(response.data)) {
        console.error('Properties response is not an array:', response.data)
        throw new Error('Invalid properties response format')
      }
      
      return response.data
    },
    {
      refetchOnWindowFocus: false,
      onError: (error) => {
        console.error('Properties fetch error:', error)
      }
    }
  )

  // Calculate average price for icon coloring
  const avgPrice = properties.reduce((sum, prop) => sum + (prop.price || 0), 0) / properties.length || 0

  // City centers for quick navigation
  const cityCenters: Record<string, [number, number]> = {
    'Sofia': [42.6977, 23.3219],
    'Plovdiv': [42.1354, 24.7453],
    'Varna': [43.2141, 27.9147],
    'Burgas': [42.5048, 27.4626],
    'Ruse': [43.8564, 25.9707],
    'Stara Zagora': [42.4258, 25.6347],
  }

  const handleCitySelect = (city: string) => {
    setSelectedCity(city)
    if (cityCenters[city]) {
      setMapCenter(cityCenters[city])
    }
  }

  const formatPrice = (price: number) => {
    return new Intl.NumberFormat('bg-BG', {
      style: 'currency',
      currency: 'BGN',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(price)
  }

  const formatFeatures = (features: string[] | undefined) => {
    if (!features || features.length === 0) return 'No features listed'
    return features.slice(0, 3).map(f => f.charAt(0).toUpperCase() + f.slice(1)).join(', ') + 
           (features.length > 3 ? ` +${features.length - 3} more` : '')
  }

  return (
    <div className="relative h-full w-full">
      {/* City selector */}
      <div className="absolute top-4 left-4 z-[1000] bg-white rounded-lg shadow-lg border border-secondary-200">
        <select
          value={selectedCity}
          onChange={(e) => handleCitySelect(e.target.value)}
          className="px-3 py-2 text-sm border-0 rounded-lg focus:ring-2 focus:ring-primary-500 focus:outline-none"
        >
          <option value="">All Cities</option>
          {Object.keys(cityCenters).map(city => (
            <option key={city} value={city}>{city}</option>
          ))}
        </select>
      </div>

      {/* Loading indicator */}
      {isLoading && (
        <div className="absolute top-4 right-4 z-[1000] bg-white rounded-lg shadow-lg border border-secondary-200 px-3 py-2">
          <div className="flex items-center space-x-2">
            <div className="spinner"></div>
            <span className="text-sm text-secondary-600">Loading properties...</span>
          </div>
        </div>
      )}

      {/* Property count */}
      <div className="absolute bottom-4 left-4 z-[1000] bg-white rounded-lg shadow-lg border border-secondary-200 px-3 py-2">
        <div className="flex items-center space-x-2 text-sm">
          <Home className="h-4 w-4 text-primary-600" />
          <span className="text-secondary-900 font-medium">
            {properties.length} {properties.length === 1 ? 'Property' : 'Properties'}
            {error && <span className="text-red-500 ml-2">(Error loading)</span>}
            {isLoading && <span className="text-blue-500 ml-2">(Loading...)</span>}
          </span>
        </div>
        {/* Debug info */}
        {process.env.NODE_ENV === 'development' && (
          <div className="text-xs text-gray-500 mt-1">
            Debug: API={API_BASE_URL}, City={selectedCity || 'All'}
          </div>
        )}
      </div>

      {/* Legend */}
      <div className="absolute bottom-4 right-4 z-[1000] bg-white rounded-lg shadow-lg border border-secondary-200 p-3">
        <h4 className="text-sm font-medium text-secondary-900 mb-2">Price Legend</h4>
        <div className="space-y-1 text-xs">
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-success-500 rounded-full"></div>
            <span className="text-secondary-700">Below average</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-primary-500 rounded-full"></div>
            <span className="text-secondary-700">Average price</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-error-500 rounded-full"></div>
            <span className="text-secondary-700">Above average</span>
          </div>
        </div>
      </div>

      <MapContainer
        center={mapCenter}
        zoom={7}
        className="h-full w-full rounded-lg"
        scrollWheelZoom={true}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        
        <MapCenter center={mapCenter} />
        
        {properties
          .filter(property => property.latitude && property.longitude && property.price)
          .map((property, index) => (
            <Marker
              key={`${property.id || index}`}
              position={[property.latitude!, property.longitude!]}
              icon={createPropertyIcon(property.price!, avgPrice)}
            >
              <Popup className="custom-popup" maxWidth={300}>
                <div className="p-2">
                  <div className="flex items-start justify-between mb-3">
                    <div>
                      <h3 className="font-semibold text-secondary-900 text-base">
                        {property.city}{property.neighborhood && `, ${property.neighborhood}`}
                      </h3>
                      <div className="flex items-center text-sm text-secondary-600 mt-1">
                        <MapPin className="h-3 w-3 mr-1" />
                        {property.latitude?.toFixed(4)}, {property.longitude?.toFixed(4)}
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-lg font-bold text-primary-600">
                        {formatPrice(property.price!)}
                      </div>
                      {property.size && (
                        <div className="text-sm text-secondary-600">
                          {formatPrice(property.price! / property.size)} / m²
                        </div>
                      )}
                    </div>
                  </div>

                  <div className="grid grid-cols-2 gap-4 mb-3 text-sm">
                    <div>
                      <span className="text-secondary-600">Size:</span>
                      <span className="ml-1 font-medium text-secondary-900">
                        {property.size ? `${property.size} m²` : 'N/A'}
                      </span>
                    </div>
                    <div>
                      <span className="text-secondary-600">Rooms:</span>
                      <span className="ml-1 font-medium text-secondary-900">
                        {property.rooms || 'N/A'}
                      </span>
                    </div>
                    <div>
                      <span className="text-secondary-600">Floor:</span>
                      <span className="ml-1 font-medium text-secondary-900">
                        {property.floor || 'N/A'}
                      </span>
                    </div>
                    <div>
                      <span className="text-secondary-600">Built:</span>
                      <span className="ml-1 font-medium text-secondary-900">
                        {property.year_built || 'N/A'}
                      </span>
                    </div>
                  </div>

                  {property.features && property.features.length > 0 && (
                    <div className="mb-3">
                      <div className="text-sm text-secondary-600 mb-1">Features:</div>
                      <div className="text-sm text-secondary-900">
                        {formatFeatures(property.features)}
                      </div>
                    </div>
                  )}

                  {property.description && (
                    <div className="mb-3">
                      <div className="text-sm text-secondary-600 mb-1">Description:</div>
                      <div className="text-sm text-secondary-900 line-clamp-2">
                        {property.description.substring(0, 100)}
                        {property.description.length > 100 && '...'}
                      </div>
                    </div>
                  )}

                  {property.url && (
                    <div className="pt-2 border-t border-secondary-200">
                      <a
                        href={property.url}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="inline-flex items-center text-primary-600 hover:text-primary-700 text-sm font-medium"
                      >
                        <Maximize className="h-3 w-3 mr-1" />
                        View Original Listing
                      </a>
                    </div>
                  )}
                </div>
              </Popup>
            </Marker>
          ))}
      </MapContainer>
    </div>
  )
}

export default PropertyMap 