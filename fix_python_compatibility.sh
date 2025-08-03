#!/bin/bash

# Quick fix for Python 3.13 compatibility issues
# Specifically addresses pandas compilation errors

set -e

echo "🔧 Python 3.13 Compatibility Fix"
echo "================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check Python version
PYTHON_VERSION=$(python3 --version | cut -d " " -f 2)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d "." -f 2)

print_status "Detected Python $PYTHON_VERSION"

if [ "$PYTHON_MINOR" -eq 13 ]; then
    print_status "Applying Python 3.13 compatibility fixes..."
    
    cd backend
    
    # Remove existing virtual environment if it exists
    if [ -d "venv" ]; then
        print_status "Removing existing virtual environment..."
        rm -rf venv
    fi
    
    # Create new virtual environment
    print_status "Creating new virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    
    # Upgrade pip and build tools first
    print_status "Upgrading pip and build tools..."
    pip install --upgrade pip setuptools wheel
    
    # Install packages with Python 3.13 compatibility
    print_status "Installing packages step by step for maximum compatibility..."
    
    # Method 1: Try installing from PyPI with latest versions
    print_status "Attempting to install core scientific packages..."
    
    # Install packages that are most likely to have Python 3.13 wheels
    print_status "Installing basic packages..."
    pip install requests==2.31.0 python-dotenv==1.0.0 httpx==0.25.2
    pip install python-multipart==0.0.6 pydantic==2.5.0 aiofiles==23.2.1
    
    # Try to install numpy and pandas with pre-compiled wheels
    print_status "Installing numpy (may take a few minutes)..."
    pip install --only-binary=all numpy || pip install numpy
    
    print_status "Installing pandas (may take a few minutes)..."
    pip install --only-binary=all pandas || pip install pandas
    
    # Install web framework packages
    print_status "Installing web framework packages..."
    pip install fastapi==0.104.1 uvicorn[standard]==0.24.0
    
    # Install HTML parsing (using html5lib instead of lxml for Python 3.13)
    print_status "Installing HTML parsing libraries..."
    pip install beautifulsoup4==4.12.2 html5lib==1.1
    
    # Try ML packages
    print_status "Installing machine learning packages..."
    pip install --only-binary=all scikit-learn || pip install scikit-learn
    pip install joblib
    
    # Try XGBoost
    print_status "Installing XGBoost..."
    pip install --only-binary=all xgboost || pip install xgboost || print_status "XGBoost failed, continuing without it..."
    
    # Install other packages
    print_status "Installing remaining packages..."
    pip install sqlalchemy==2.0.23 geopy==2.4.0
    pip install --only-binary=all matplotlib || pip install matplotlib
    pip install --only-binary=all plotly || pip install plotly
    pip install pytest
    
    print_success "All packages installed successfully!"
    
    cd ..
    
else
    print_status "Python $PYTHON_VERSION should work with standard requirements."
    print_status "Running standard setup..."
    ./setup.sh
fi

print_success "Compatibility fix completed!"
echo ""
echo "Next steps:"
echo "1. Run: ./generate_sample_data.sh"
echo "2. Run: ./start_all.sh"
echo "3. Open: http://localhost:3000"
echo ""
echo "If you still have issues, check TROUBLESHOOTING.md" 