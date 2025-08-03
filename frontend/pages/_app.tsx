import type { AppProps } from 'next/app'
import { QueryClient, QueryClientProvider } from 'react-query'
import { Toaster } from 'react-hot-toast'
import Head from 'next/head'
import '../styles/globals.css'

// Create a client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
      staleTime: 5 * 60 * 1000, // 5 minutes
    },
  },
})

export default function App({ Component, pageProps }: AppProps) {
  return (
    <>
      <Head>
        <title>Bulgarian Real Estate Price Predictor</title>
        <meta name="description" content="Predict Bulgarian property prices using machine learning with location, size, and features analysis" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <link rel="icon" href="/favicon.ico" />
        <meta property="og:title" content="Bulgarian Real Estate Price Predictor" />
        <meta property="og:description" content="AI-powered property price predictions for the Bulgarian market" />
        <meta property="og:type" content="website" />
        <meta name="twitter:card" content="summary_large_image" />
        <meta name="twitter:title" content="Bulgarian Real Estate Price Predictor" />
        <meta name="twitter:description" content="AI-powered property price predictions for the Bulgarian market" />
      </Head>
      
      <QueryClientProvider client={queryClient}>
        <Component {...pageProps} />
        <Toaster 
          position="top-right"
          toastOptions={{
            duration: 4000,
            style: {
              background: '#1f2937',
              color: '#f9fafb',
              borderRadius: '0.5rem',
              border: '1px solid #374151',
            },
            success: {
              iconTheme: {
                primary: '#10b981',
                secondary: '#f9fafb',
              },
            },
            error: {
              iconTheme: {
                primary: '#ef4444',
                secondary: '#f9fafb',
              },
            },
          }}
        />
      </QueryClientProvider>
    </>
  )
} 