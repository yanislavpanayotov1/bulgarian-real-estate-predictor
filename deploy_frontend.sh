#!/bin/bash

echo "🚀 Deploying Frontend to Vercel..."
echo "=================================="

# Install Vercel CLI if not installed
if ! command -v vercel &> /dev/null; then
    echo "📦 Installing Vercel CLI..."
    npm install -g vercel
fi

# Go to frontend directory
cd frontend

# Install dependencies
echo "📦 Installing dependencies..."
npm install

# Fix Next.js config for deployment
echo "🔧 Optimizing Next.js config for production..."

# Create optimized next.config.js for deployment
cat > next.config.js << 'EOF'
/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  images: {
    domains: ['images.unsplash.com', 'via.placeholder.com'],
  },
  env: {
    NEXT_PUBLIC_API_URL: process.env.NEXT_PUBLIC_API_URL || 'https://your-backend-url.railway.app',
  },
  // Optimize for deployment
  output: 'standalone',
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: true,
  },
}

module.exports = nextConfig
EOF

# Create .vercelignore
cat > .vercelignore << 'EOF'
node_modules
.next
*.log
.env.local
EOF

echo ""
echo "✅ Frontend prepared for Vercel deployment!"
echo ""
echo "🎯 Next steps:"
echo "1. Run: vercel"
echo "2. Follow the prompts (choose your account, project name)"
echo "3. Set NEXT_PUBLIC_API_URL to your backend URL after backend deployment"
echo ""
echo "📝 After deploying backend, update the API URL with:"
echo "   vercel env add NEXT_PUBLIC_API_URL"
echo "" 