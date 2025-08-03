#!/bin/bash

# One-Command Fix for All Python 3.13 Issues
# This script will completely fix your setup and get the project running

set -e

echo "🔧 Complete Python 3.13 Fix & Setup"
echo "===================================="
echo ""
echo "This will completely fix all compatibility issues and set up your project."
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${BLUE}[INFO]${NC} $1"; }
print_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Step 1: Clean existing setup
print_status "Step 1: Cleaning existing setup..."
if [ -d "backend/venv" ]; then
    print_status "Removing old virtual environment..."
    rm -rf backend/venv
fi

if [ -d "frontend/node_modules" ]; then
    print_status "Removing old node_modules..."
    rm -rf frontend/node_modules
fi

if [ -d "frontend/.next" ]; then
    rm -rf frontend/.next
fi

# Step 2: Setup Python backend with compatibility fixes
print_status "Step 2: Setting up Python backend..."
cd backend

# Create new virtual environment
print_status "Creating new virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install packages with Python 3.13 compatibility
print_status "Installing packages with Python 3.13 compatibility..."

# Essential packages (no compilation needed)
print_status "Installing essential packages..."
pip install requests python-dotenv httpx python-multipart pydantic aiofiles pytest

# Web framework
print_status "Installing web framework..."
pip install fastapi==0.104.1 uvicorn[standard]==0.24.0

# HTML parsing (using html5lib instead of lxml)
print_status "Installing HTML parsing libraries..."
pip install beautifulsoup4 html5lib

# Database
print_status "Installing database libraries..."
pip install sqlalchemy

# Geolocation
print_status "Installing geolocation libraries..."
pip install geopy

# Try to install scientific packages
print_status "Installing scientific packages (may take a few minutes)..."

# Try numpy with fallback
if ! pip install --only-binary=all "numpy>=1.26.0" 2>/dev/null; then
    print_warning "Pre-compiled numpy not available, trying source installation..."
    pip install numpy || print_error "numpy installation failed"
fi

# Try pandas with fallback
if ! pip install --only-binary=all "pandas>=2.2.0" 2>/dev/null; then
    print_warning "Pre-compiled pandas not available, trying source installation..."
    pip install pandas || print_error "pandas installation failed"
fi

# Try scikit-learn
if ! pip install --only-binary=all "scikit-learn>=1.4.0" 2>/dev/null; then
    print_warning "Pre-compiled scikit-learn not available, trying source installation..."
    pip install scikit-learn || print_error "scikit-learn installation failed"
fi

# Install joblib (needed for model persistence)
pip install joblib

# Try ML packages (optional)
print_status "Installing optional ML packages..."
pip install --only-binary=all xgboost 2>/dev/null || print_warning "XGBoost not available (optional)"

# Try visualization packages (optional)
print_status "Installing optional visualization packages..."
pip install --only-binary=all matplotlib 2>/dev/null || print_warning "matplotlib not available (optional)"
pip install --only-binary=all plotly 2>/dev/null || print_warning "plotly not available (optional)"
pip install --only-binary=all seaborn 2>/dev/null || print_warning "seaborn not available (optional)"

print_success "Backend packages installed!"

# Create directories
mkdir -p ../data ../models

cd ..

# Step 3: Setup frontend
print_status "Step 3: Setting up frontend..."
cd frontend

# Check if Node.js is available
if ! command -v node &> /dev/null; then
    print_error "Node.js is not installed. Please install Node.js 16+ and run this script again."
    exit 1
fi

print_status "Installing frontend dependencies..."
npm install

print_success "Frontend packages installed!"

cd ..

# Step 4: Create environment files
print_status "Step 4: Creating environment configuration..."

# Backend .env
cat > backend/.env << 'EOF'
# Backend Configuration
PYTHONPATH=.
LOG_LEVEL=INFO

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Data paths
DATA_PATH=../data
MODELS_PATH=../models
EOF

# Frontend .env.local
cat > frontend/.env.local << 'EOF'
# Frontend Configuration
NEXT_PUBLIC_API_URL=http://localhost:8000
EOF

print_success "Environment files created!"

# Step 5: Test the setup
print_status "Step 5: Testing the setup..."

cd backend
source venv/bin/activate

python3 -c "
try:
    import pandas, numpy, sklearn, fastapi, requests, bs4
    print('✅ Core packages working')
    
    # Test our ML model
    import sys
    sys.path.append('.')
    from ml_model.train_model import RealEstatePricePredictor
    predictor = RealEstatePricePredictor()
    sample_data = predictor._create_sample_data()
    print(f'✅ ML model working - generated {len(sample_data)} sample properties')
    
    # Save sample data
    sample_data.to_csv('../data/raw_properties.csv', index=False)
    print('✅ Sample data saved')
    
except Exception as e:
    print(f'❌ Setup test failed: {e}')
    exit(1)
"

cd ..

print_success "Setup test passed!"

# Step 6: Create startup script
print_status "Step 6: Creating startup scripts..."

cat > start_app.sh << 'EOF'
#!/bin/bash
echo "🏠 Starting Bulgarian Real Estate Price Predictor..."
echo ""

# Start backend
echo "📊 Starting backend API server..."
cd backend
source venv/bin/activate
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
cd ..

# Wait for backend to start
sleep 5

# Start frontend
echo "🎨 Starting frontend web application..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo ""
echo "✅ Application started successfully!"
echo ""
echo "🌐 URLs:"
echo "   Frontend:    http://localhost:3000"
echo "   Backend API: http://localhost:8000"
echo "   API Docs:    http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop both services"

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Stopping services..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    exit 0
}

trap cleanup SIGINT SIGTERM

# Wait for processes
wait $BACKEND_PID $FRONTEND_PID
EOF

chmod +x start_app.sh

print_success "Startup script created!"

echo ""
echo "🎉 COMPLETE SETUP FINISHED!"
echo "=========================="
echo ""
echo "Everything is now working! Here's what was fixed:"
echo ""
echo "✅ Removed problematic packages (lxml, selenium)"
echo "✅ Updated to Python 3.13 compatible versions"
echo "✅ Used html5lib instead of lxml for HTML parsing"
echo "✅ Installed packages with fallback strategies"
echo "✅ Created sample data for testing"
echo "✅ Set up both backend and frontend"
echo ""
echo "🚀 TO START THE APPLICATION:"
echo "   ./start_app.sh"
echo ""
echo "Then open: http://localhost:3000"
echo ""
print_success "All issues fixed! Ready to use! 🎯" 