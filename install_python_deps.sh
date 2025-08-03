#!/bin/bash

# Comprehensive Python Dependency Installer
# Handles Python 3.9-3.13 with fallback strategies

set -e

echo "🐍 Python Dependency Installer"
echo "=============================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

# Check Python version and set strategy
detect_python_strategy() {
    PYTHON_VERSION=$(python3 --version | cut -d " " -f 2)
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d "." -f 1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d "." -f 2)
    
    print_status "Detected Python $PYTHON_VERSION"
    
    if [ "$PYTHON_MAJOR" -eq 3 ]; then
        if [ "$PYTHON_MINOR" -eq 13 ]; then
            STRATEGY="python313"
            print_warning "Python 3.13 detected - using compatibility strategy"
        elif [ "$PYTHON_MINOR" -eq 12 ]; then
            STRATEGY="python312"
            print_status "Python 3.12 detected - using standard strategy"
        elif [ "$PYTHON_MINOR" -eq 11 ]; then
            STRATEGY="python311"
            print_status "Python 3.11 detected - using standard strategy"
        elif [ "$PYTHON_MINOR" -eq 10 ]; then
            STRATEGY="python310"
            print_status "Python 3.10 detected - using legacy strategy"
        elif [ "$PYTHON_MINOR" -eq 9 ]; then
            STRATEGY="python39"
            print_warning "Python 3.9 detected - using legacy strategy"
        else
            STRATEGY="unknown"
            print_error "Unsupported Python version: $PYTHON_VERSION"
            exit 1
        fi
    else
        print_error "Python 3 is required. Found Python $PYTHON_VERSION"
        exit 1
    fi
}

# Install packages with fallback strategies
install_with_fallback() {
    local package=$1
    local alternatives=$2
    
    print_status "Installing $package..."
    
    # Try pre-compiled wheels first
    if pip install --only-binary=all "$package" 2>/dev/null; then
        print_success "$package installed (binary wheel)"
        return 0
    fi
    
    # Try standard installation
    if pip install "$package" 2>/dev/null; then
        print_success "$package installed (source)"
        return 0
    fi
    
    # Try alternatives if provided
    if [ -n "$alternatives" ]; then
        print_warning "$package failed, trying alternatives..."
        IFS=',' read -ra ALTS <<< "$alternatives"
        for alt in "${ALTS[@]}"; do
            if pip install "$alt" 2>/dev/null; then
                print_success "$alt installed as alternative to $package"
                return 0
            fi
        done
    fi
    
    print_error "Failed to install $package"
    return 1
}

# Python 3.13 specific installation
install_python313() {
    print_status "Using Python 3.13 compatibility strategy..."
    
    # Essential packages first
    print_status "Installing essential packages..."
    pip install requests python-dotenv httpx python-multipart pydantic aiofiles
    
    # Scientific packages with specific versions
    print_status "Installing scientific packages..."
    install_with_fallback "numpy>=1.26.0" ""
    install_with_fallback "pandas>=2.2.0" ""
    
    # Web framework
    print_status "Installing web framework..."
    pip install fastapi==0.104.1 uvicorn[standard]==0.24.0
    
    # HTML parsing (avoid lxml on 3.13)
    print_status "Installing HTML parsing..."
    pip install beautifulsoup4 html5lib
    
    # ML packages
    print_status "Installing ML packages..."
    install_with_fallback "scikit-learn>=1.4.0" ""
    install_with_fallback "joblib" ""
    install_with_fallback "xgboost>=2.0.3" ""
    
    # Other packages
    print_status "Installing additional packages..."
    pip install sqlalchemy geopy
    install_with_fallback "matplotlib" ""
    install_with_fallback "plotly" ""
    install_with_fallback "seaborn" ""
    pip install pytest
}

# Standard installation for Python 3.11-3.12
install_standard() {
    print_status "Using standard installation strategy..."
    
    # Install from requirements.txt
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
    else
        print_error "requirements.txt not found"
        exit 1
    fi
}

# Legacy installation for Python 3.9-3.10
install_legacy() {
    print_status "Using legacy installation strategy..."
    
    # Use older package versions
    pip install fastapi==0.95.0 uvicorn[standard]==0.20.0
    pip install pandas==1.5.3 numpy==1.24.3
    pip install scikit-learn==1.2.2 xgboost==1.7.5
    pip install requests beautifulsoup4 html5lib
    pip install python-multipart pydantic==1.10.12
    pip install sqlalchemy==1.4.48 python-dotenv geopy
    pip install matplotlib plotly seaborn joblib
    pip install httpx pytest aiofiles
}

# Check if we're in a virtual environment
check_venv() {
    if [ -z "$VIRTUAL_ENV" ]; then
        print_error "Please activate a virtual environment first!"
        echo "Run: source venv/bin/activate"
        exit 1
    fi
    print_status "Virtual environment: $VIRTUAL_ENV"
}

# Main installation function
main() {
    check_venv
    detect_python_strategy
    
    # Upgrade pip first
    print_status "Upgrading pip..."
    pip install --upgrade pip setuptools wheel
    
    # Install based on strategy
    case $STRATEGY in
        "python313")
            install_python313
            ;;
        "python312"|"python311")
            install_standard
            ;;
        "python310"|"python39")
            install_legacy
            ;;
        *)
            print_error "Unknown strategy: $STRATEGY"
            exit 1
            ;;
    esac
    
    print_success "All dependencies installed successfully!"
    
    # Verify installation
    print_status "Verifying installation..."
    python3 -c "
import pandas, numpy, sklearn, fastapi, requests, bs4
print('✅ Core packages imported successfully')
try:
    import xgboost
    print('✅ XGBoost available')
except ImportError:
    print('⚠️  XGBoost not available (optional)')
try:
    import matplotlib, plotly, seaborn
    print('✅ Visualization packages available')
except ImportError:
    print('⚠️  Some visualization packages missing (optional)')
"
    
    echo ""
    print_success "Installation completed!"
    echo ""
    echo "Next steps:"
    echo "1. Run: python3 ../test_setup.py"
    echo "2. Run: ../generate_sample_data.sh"
    echo "3. Run: ../start_all.sh"
}

# Run main function
main "$@" 