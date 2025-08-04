import { useState, useEffect } from 'react'
import { useQuery, useMutation } from 'react-query'
import { useForm } from 'react-hook-form'
import { zodResolver } from '@hookform/resolvers/zod'
import { z } from 'zod'
import axios from 'axios'
import toast from 'react-hot-toast'
import dynamic from 'next/dynamic'
import { 
  Home, 
  MapPin, 
  Calculator, 
  TrendingUp, 
  Building2, 
  DollarSign, 
  BarChart3,
  Loader2,
  Search,
  Filter,
  Info
} from 'lucide-react'

// Dynamic import for map component to avoid SSR issues
const PropertyMap = dynamic(() => import('../components/PropertyMap'), {
  ssr: false,
  loading: () => (
    <div className="h-96 bg-secondary-100 rounded-xl flex items-center justify-center">
      <Loader2 className="h-8 w-8 animate-spin text-primary-600" />
    </div>
  )
})

// API base URL
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

// Validation schema
const propertySchema = z.object({
  size: z.number().min(10, 'Size must be at least 10 m²').max(500, 'Size cannot exceed 500 m²'),
  rooms: z.number().min(1, 'Must have at least 1 room').max(10, 'Cannot exceed 10 rooms'),
  floor: z.number().min(1, 'Floor must be at least 1').max(50, 'Floor cannot exceed 50'),
  year_built: z.number().min(1800, 'Year must be at least 1800').max(2024, 'Year cannot exceed 2024'),
  city: z.string().min(1, 'City is required'),
  neighborhood: z.string().optional(),
  latitude: z.number().optional(),
  longitude: z.number().optional(),
  features: z.array(z.string()).optional(),
})

type PropertyFormData = z.infer<typeof propertySchema>

interface PredictionResult {
  predicted_price: number
  confidence_lower: number
  confidence_upper: number
  price_per_sqm: number
  model_used: string
  prediction_timestamp: string
}

interface MarketStats {
  avg_price: number
  avg_price_per_sqm: number
  avg_size: number
  total_properties: number
  price_range: {
    min: number
    max: number
    q25: number
    q75: number
  }
  popular_cities: Array<{
    city: string
    count: number
    avg_price: number
  }>
  property_age_distribution: Record<string, number>
}

const BULGARIAN_CITIES = [
  'Sofia', 'Plovdiv', 'Varna', 'Burgas', 'Ruse', 'Stara Zagora', 
  'Pleven', 'Sliven', 'Dobrich', 'Shumen', 'Pernik', 'Haskovo'
]

const PROPERTY_FEATURES = [
  'parking', 'balcony', 'elevator', 'terrace', 'air conditioning',
  'heating', 'furnished', 'garage', 'garden', 'swimming pool'
]

export default function HomePage() {
  const [prediction, setPrediction] = useState<PredictionResult | null>(null)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [selectedFeatures, setSelectedFeatures] = useState<string[]>([])

  const {
    register,
    handleSubmit,
    formState: { errors },
    setValue,
    watch,
    reset
  } = useForm<PropertyFormData>({
    resolver: zodResolver(propertySchema),
    defaultValues: {
      city: 'Sofia',
      features: []
    }
  })

  // Fetch market statistics
  const { data: marketStats, isLoading: statsLoading } = useQuery<MarketStats>(
    'market-stats',
    async () => {
      const response = await axios.get(`${API_BASE_URL}/market-stats`)
      return response.data
    }
  )

  // Prediction mutation
  const predictMutation = useMutation(
    async (data: PropertyFormData) => {
      console.log('Making prediction request to:', `${API_BASE_URL}/predict`)
      console.log('Prediction payload:', { ...data, features: selectedFeatures })
      
      const response = await axios.post(`${API_BASE_URL}/predict`, {
        ...data,
        features: selectedFeatures
      })
      
      console.log('Prediction response:', response.data)
      return response.data
    },
    {
      onSuccess: (data) => {
        console.log('Prediction success:', data)
        setPrediction(data)
        toast.success('Price prediction generated successfully!')
      },
      onError: (error: any) => {
        console.error('Prediction error:', error)
        console.error('Error response:', error.response?.data)
        toast.error(error.response?.data?.detail || 'Prediction failed')
      }
    }
  )

  const handlePredict = (data: PropertyFormData) => {
    predictMutation.mutate({
      ...data,
      features: selectedFeatures
    })
  }

  const handleFeatureToggle = (feature: string) => {
    setSelectedFeatures(prev => 
      prev.includes(feature) 
        ? prev.filter(f => f !== feature)
        : [...prev, feature]
    )
  }

  const formatPrice = (price: number) => {
    return new Intl.NumberFormat('de-DE', {
      style: 'currency',
      currency: 'EUR',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(price)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-secondary-50 to-secondary-100">
      {/* Header */}
      <header className="bg-white shadow-soft border-b border-secondary-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="gradient-bg p-3 rounded-xl">
                <Home className="h-8 w-8 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-secondary-900">
                  Bulgarian Real Estate Predictor
                </h1>
                <p className="text-secondary-600 text-sm">
                  AI-powered property price predictions
                </p>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <div className="status-online">
                <div className="w-2 h-2 bg-success-500 rounded-full mr-2"></div>
                API Online
              </div>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Market Stats Overview */}
        {marketStats && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <div className="stats-card">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-secondary-600">Avg Price</p>
                  <p className="text-2xl font-bold text-secondary-900">
                    {formatPrice(marketStats.avg_price)}
                  </p>
                </div>
                <DollarSign className="h-8 w-8 text-primary-600" />
              </div>
            </div>

            <div className="stats-card">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-secondary-600">Price per m²</p>
                  <p className="text-2xl font-bold text-secondary-900">
                    {formatPrice(marketStats.avg_price_per_sqm)}
                  </p>
                </div>
                <BarChart3 className="h-8 w-8 text-primary-600" />
              </div>
            </div>

            <div className="stats-card">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-secondary-600">Avg Size</p>
                  <p className="text-2xl font-bold text-secondary-900">
                    {Math.round(marketStats.avg_size)} m²
                  </p>
                </div>
                <Building2 className="h-8 w-8 text-primary-600" />
              </div>
            </div>

            <div className="stats-card">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-medium text-secondary-600">Properties</p>
                  <p className="text-2xl font-bold text-secondary-900">
                    {marketStats.total_properties.toLocaleString()}
                  </p>
                </div>
                <TrendingUp className="h-8 w-8 text-primary-600" />
              </div>
            </div>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Prediction Form */}
          <div className="lg:col-span-1">
            <div className="card">
              <div className="card-header">
                <h2 className="text-xl font-semibold text-secondary-900 flex items-center">
                  <Calculator className="h-5 w-5 mr-2 text-primary-600" />
                  Price Prediction
                </h2>
              </div>

              <form onSubmit={handleSubmit(handlePredict)} className="space-y-6">
                {/* Basic Fields */}
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="label-text">Size (m²)</label>
                    <input
                      type="number"
                      {...register('size', { valueAsNumber: true })}
                      className="input-field"
                      placeholder="75"
                    />
                    {errors.size && (
                      <p className="error-message">{errors.size.message}</p>
                    )}
                  </div>

                  <div>
                    <label className="label-text">Rooms</label>
                    <input
                      type="number"
                      {...register('rooms', { valueAsNumber: true })}
                      className="input-field"
                      placeholder="2"
                    />
                    {errors.rooms && (
                      <p className="error-message">{errors.rooms.message}</p>
                    )}
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="label-text">Floor</label>
                    <input
                      type="number"
                      {...register('floor', { valueAsNumber: true })}
                      className="input-field"
                      placeholder="3"
                    />
                    {errors.floor && (
                      <p className="error-message">{errors.floor.message}</p>
                    )}
                  </div>

                  <div>
                    <label className="label-text">Year Built</label>
                    <input
                      type="number"
                      {...register('year_built', { valueAsNumber: true })}
                      className="input-field"
                      placeholder="2010"
                    />
                    {errors.year_built && (
                      <p className="error-message">{errors.year_built.message}</p>
                    )}
                  </div>
                </div>

                <div>
                  <label className="label-text">City</label>
                  <select {...register('city')} className="input-field">
                    {BULGARIAN_CITIES.map(city => (
                      <option key={city} value={city}>{city}</option>
                    ))}
                  </select>
                  {errors.city && (
                    <p className="error-message">{errors.city.message}</p>
                  )}
                </div>

                {/* Advanced Fields Toggle */}
                <div>
                  <button
                    type="button"
                    onClick={() => setShowAdvanced(!showAdvanced)}
                    className="flex items-center text-primary-600 hover:text-primary-700 font-medium"
                  >
                    {showAdvanced ? 'Hide' : 'Show'} Advanced Options
                    <Filter className="h-4 w-4 ml-1" />
                  </button>
                </div>

                {showAdvanced && (
                  <div className="space-y-4 border-t border-secondary-200 pt-4">
                    <div>
                      <label className="label-text">Neighborhood (optional)</label>
                      <input
                        type="text"
                        {...register('neighborhood')}
                        className="input-field"
                        placeholder="Center, Mladost, Vitosha..."
                      />
                    </div>

                    <div>
                      <label className="label-text">Features & Amenities</label>
                      <div className="grid grid-cols-2 gap-2 mt-2">
                        {PROPERTY_FEATURES.map(feature => (
                          <label key={feature} className="flex items-center space-x-2">
                            <input
                              type="checkbox"
                              checked={selectedFeatures.includes(feature)}
                              onChange={() => handleFeatureToggle(feature)}
                              className="rounded border-secondary-300 text-primary-600 focus:ring-primary-500"
                            />
                            <span className="text-sm text-secondary-700 capitalize">
                              {feature}
                            </span>
                          </label>
                        ))}
                      </div>
                    </div>
                  </div>
                )}

                <button
                  type="submit"
                  disabled={predictMutation.isLoading}
                  className="btn-primary w-full flex items-center justify-center"
                >
                  {predictMutation.isLoading ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin mr-2" />
                      Predicting...
                    </>
                  ) : (
                    <>
                      <Calculator className="h-4 w-4 mr-2" />
                      Predict Price
                    </>
                  )}
                </button>
              </form>

              {/* Prediction Result */}
              {prediction && (
                <div className="mt-6 p-4 bg-primary-50 border border-primary-200 rounded-lg">
                  <h3 className="font-semibold text-primary-900 mb-3">
                    Price Prediction Result
                  </h3>
                  
                  <div className="space-y-2">
                    <div className="flex justify-between items-center">
                      <span className="text-primary-700">Predicted Price:</span>
                      <span className="price-highlight">
                        {formatPrice(prediction.predicted_price)}
                      </span>
                    </div>
                    
                    <div className="flex justify-between items-center">
                      <span className="text-primary-700">Price per m²:</span>
                      <span className="font-medium text-primary-900">
                        {formatPrice(prediction.price_per_sqm)}
                      </span>
                    </div>
                    
                    <div className="text-xs text-primary-600 mt-2">
                      Confidence Range: {formatPrice(prediction.confidence_lower)} - {formatPrice(prediction.confidence_upper)}
                    </div>
                    
                    <div className="text-xs text-primary-600">
                      Model: {prediction.model_used}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Map and Analytics */}
          <div className="lg:col-span-2 space-y-8">
            {/* Interactive Map */}
            <div className="card">
              <div className="card-header">
                <h2 className="text-xl font-semibold text-secondary-900 flex items-center">
                  <MapPin className="h-5 w-5 mr-2 text-primary-600" />
                  Property Map
                </h2>
                <div className="flex items-center text-sm text-secondary-600">
                  <Info className="h-4 w-4 mr-1" />
                  Click markers for details
                </div>
              </div>
              
              <div className="h-96">
                <PropertyMap />
              </div>
            </div>

            {/* Popular Cities */}
            {marketStats?.popular_cities && (
              <div className="card">
                <div className="card-header">
                  <h2 className="text-xl font-semibold text-secondary-900">
                    Popular Cities
                  </h2>
                </div>
                
                <div className="space-y-3">
                  {marketStats.popular_cities.slice(0, 6).map((city, index) => (
                    <div key={city.city} className="flex items-center justify-between py-2">
                      <div className="flex items-center space-x-3">
                        <div className="w-8 h-8 bg-primary-100 text-primary-600 rounded-full flex items-center justify-center text-sm font-medium">
                          {index + 1}
                        </div>
                        <div>
                          <p className="font-medium text-secondary-900">{city.city}</p>
                          <p className="text-sm text-secondary-600">
                            {city.count} properties
                          </p>
                        </div>
                      </div>
                      <div className="text-right">
                        <p className="font-medium text-secondary-900">
                          {formatPrice(city.avg_price)}
                        </p>
                        <p className="text-sm text-secondary-600">avg. price</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
} 