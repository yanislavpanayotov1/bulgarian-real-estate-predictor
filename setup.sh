#!/bin/bash

# Bulgarian Real Estate Price Predictor Setup Script
# This script sets up both the backend (Python) and frontend (Node.js) components

set -e  # Exit on any error

echo "🏠 Bulgarian Real Estate Price Predictor Setup"
echo "=============================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is installed
check_python() {
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d " " -f 2)
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d "." -f 1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d "." -f 2)
        
        print_status "Python $PYTHON_VERSION found"
        
        if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -ge 11 ]; then
            if [ "$PYTHON_MINOR" -eq 13 ]; then
                print_warning "Python 3.13 detected. Using compatibility requirements."
                REQUIREMENTS_FILE="requirements.txt"
            else
                print_status "Python 3.$PYTHON_MINOR is compatible."
                REQUIREMENTS_FILE="requirements.txt"
            fi
            return 0
        else
            print_error "Python 3.11 or higher is required. Found Python $PYTHON_VERSION"
            return 1
        fi
    else
        print_error "Python 3 is not installed. Please install Python 3.11 or higher."
        return 1
    fi
}

# Check if Node.js is installed
check_node() {
    if command -v node &> /dev/null; then
        NODE_VERSION=$(node --version)
        print_status "Node.js $NODE_VERSION found"
        return 0
    else
        print_error "Node.js is not installed. Please install Node.js 16 or higher."
        return 1
    fi
}

# Setup backend
setup_backend() {
    print_status "Setting up Python backend..."
    
    cd backend
    
    # Create virtual environment
    if [ ! -d "venv" ]; then
        print_status "Creating Python virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    print_status "Activating virtual environment..."
    source venv/bin/activate
    
    # Upgrade pip
    print_status "Upgrading pip..."
    pip install --upgrade pip
    
    # Install dependencies using comprehensive installer
    print_status "Installing Python dependencies..."
    chmod +x ../install_python_deps.sh
    ../install_python_deps.sh
    
    # Create necessary directories
    mkdir -p ../data ../models
    
    print_success "Backend setup completed!"
    
    cd ..
}

# Setup frontend
setup_frontend() {
    print_status "Setting up Node.js frontend..."
    
    cd frontend
    
    # Install dependencies
    print_status "Installing Node.js dependencies..."
    npm install
    
    print_success "Frontend setup completed!"
    
    cd ..
}

# Create environment file
create_env_file() {
    print_status "Creating environment configuration..."
    
    # Backend .env
    if [ ! -f "backend/.env" ]; then
        cat > backend/.env << EOF
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
        print_success "Created backend/.env"
    fi
    
    # Frontend .env.local
    if [ ! -f "frontend/.env.local" ]; then
        cat > frontend/.env.local << EOF
# Frontend Configuration
NEXT_PUBLIC_API_URL=http://localhost:8000
EOF
        print_success "Created frontend/.env.local"
    fi
}

# Create startup scripts
create_startup_scripts() {
    print_status "Creating startup scripts..."
    
    # Backend startup script
    cat > start_backend.sh << 'EOF'
#!/bin/bash
echo "🚀 Starting Bulgarian Real Estate Backend..."
cd backend
source venv/bin/activate
echo "📊 Training ML model (this may take a few minutes)..."
python ml_model/train_model.py
echo "🌐 Starting FastAPI server..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
EOF
    chmod +x start_backend.sh
    
    # Frontend startup script  
    cat > start_frontend.sh << 'EOF'
#!/bin/bash
echo "🎨 Starting Bulgarian Real Estate Frontend..."
cd frontend
npm run dev
EOF
    chmod +x start_frontend.sh
    
    # Combined startup script
    cat > start_all.sh << 'EOF'
#!/bin/bash
echo "🏠 Starting Bulgarian Real Estate Price Predictor..."
echo "This will start both the backend API and frontend web application."
echo ""

# Start backend in background
echo "📊 Starting backend API server..."
./start_backend.sh &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Start frontend
echo "🎨 Starting frontend web application..."
./start_frontend.sh &
FRONTEND_PID=$!

echo ""
echo "✅ Application is starting up!"
echo "📊 Backend API: http://localhost:8000"
echo "🎨 Frontend Web App: http://localhost:3000"
echo "📖 API Documentation: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop both services"

# Wait for either process to exit
wait $BACKEND_PID $FRONTEND_PID
EOF
    chmod +x start_all.sh
    
    print_success "Created startup scripts"
}

# Create sample data scraper script
create_sample_data() {
    print_status "Creating sample data generation script..."
    
    cat > generate_sample_data.sh << 'EOF'
#!/bin/bash
echo "📊 Generating sample real estate data..."
cd backend
source venv/bin/activate
python -c "
from ml_model.train_model import RealEstatePricePredictor
import pandas as pd

# Create predictor and generate sample data
predictor = RealEstatePricePredictor()
df = predictor._create_sample_data()

# Save to CSV
df.to_csv('../data/raw_properties.csv', index=False)
print(f'✅ Generated {len(df)} sample properties')
print('📁 Saved to data/raw_properties.csv')
"
echo "🎯 Sample data generation completed!"
EOF
    chmod +x generate_sample_data.sh
    
    print_success "Created sample data generator"
}

# Main setup function
main() {
    echo "Starting setup process..."
    echo ""
    
    # Check prerequisites
    print_status "Checking prerequisites..."
    if ! check_python; then
        exit 1
    fi
    
    if ! check_node; then
        exit 1
    fi
    
    # Setup components
    setup_backend
    setup_frontend
    create_env_file
    create_startup_scripts
    create_sample_data
    
    echo ""
    echo "🎉 Setup completed successfully!"
    echo ""
    echo "📋 Next Steps:"
    echo "1. Generate sample data: ./generate_sample_data.sh"
    echo "2. Start the application: ./start_all.sh"
    echo "3. Open your browser to: http://localhost:3000"
    echo ""
    echo "📖 Available Scripts:"
    echo "   ./start_backend.sh    - Start only the backend API"
    echo "   ./start_frontend.sh   - Start only the frontend"
    echo "   ./start_all.sh        - Start both backend and frontend"
    echo ""
    echo "🔧 Development URLs:"
    echo "   Frontend:      http://localhost:3000"
    echo "   Backend API:   http://localhost:8000"
    echo "   API Docs:      http://localhost:8000/docs"
    echo ""
    print_success "Happy predicting! 🏠📈"
}

# Run main function
main "$@" 