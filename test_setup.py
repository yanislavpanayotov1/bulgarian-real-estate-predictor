#!/usr/bin/env python3
"""
Quick test script to verify the Bulgarian Real Estate Price Predictor setup
"""

import sys
import importlib
import subprocess
from pathlib import Path

def test_python_version():
    """Test if Python version is adequate"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Need Python 3.8+")
        return False

def test_imports():
    """Test if all required packages can be imported"""
    required_packages = [
        'fastapi',
        'uvicorn', 
        'pandas',
        'numpy',
        'sklearn',
        'xgboost',
        'requests',
        'bs4',  # beautifulsoup4 imports as bs4
        'geopy',
        'joblib'
    ]
    
    success = True
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package} - OK")
        except ImportError:
            print(f"❌ {package} - Missing")
            success = False
    
    return success

def test_directories():
    """Test if required directories exist"""
    required_dirs = [
        'backend',
        'frontend', 
        'data',
        'models'
    ]
    
    success = True
    for dir_name in required_dirs:
        path = Path(dir_name)
        if path.exists():
            print(f"✅ {dir_name}/ directory - OK")
        else:
            print(f"❌ {dir_name}/ directory - Missing")
            success = False
    
    return success

def test_files():
    """Test if key files exist"""
    required_files = [
        'backend/requirements.txt',
        'backend/api/main.py',
        'backend/ml_model/train_model.py',
        'backend/scraper/imoti_scraper.py',
        'frontend/package.json',
        'frontend/pages/index.tsx',
        'frontend/components/PropertyMap.tsx',
        'setup.sh'
    ]
    
    success = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"✅ {file_path} - OK")
        else:
            print(f"❌ {file_path} - Missing")
            success = False
    
    return success

def test_ml_model():
    """Test if the ML model can be instantiated"""
    try:
        sys.path.append('backend')
        from ml_model.train_model import RealEstatePricePredictor
        
        predictor = RealEstatePricePredictor()
        sample_data = predictor._create_sample_data()
        
        if len(sample_data) > 0:
            print(f"✅ ML Model - OK (Generated {len(sample_data)} sample properties)")
            return True
        else:
            print("❌ ML Model - Failed to generate sample data")
            return False
            
    except Exception as e:
        print(f"❌ ML Model - Error: {str(e)}")
        return False

def main():
    """Run all tests"""
    print("🏠 Bulgarian Real Estate Price Predictor - Setup Test")
    print("=" * 55)
    print()
    
    tests = [
        ("Python Version", test_python_version),
        ("Package Imports", test_imports),
        ("Directory Structure", test_directories),
        ("Required Files", test_files),
        ("ML Model", test_ml_model),
    ]
    
    all_passed = True
    
    for test_name, test_func in tests:
        print(f"Testing {test_name}...")
        try:
            success = test_func()
            if not success:
                all_passed = False
        except Exception as e:
            print(f"❌ {test_name} - Error: {str(e)}")
            all_passed = False
        print()
    
    print("=" * 55)
    if all_passed:
        print("🎉 All tests passed! Your setup is ready.")
        print()
        print("Next steps:")
        print("1. Run: ./generate_sample_data.sh")
        print("2. Run: ./start_all.sh")  
        print("3. Open: http://localhost:3000")
    else:
        print("❌ Some tests failed. Please check the setup.")
        print()
        print("Try running: ./setup.sh")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 