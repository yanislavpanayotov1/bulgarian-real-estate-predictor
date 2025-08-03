# 🔧 Troubleshooting Guide

## ⚡ Quick Fix (Recommended)

**Problem**: Multiple Python 3.13 compatibility issues (pandas, lxml, etc.)

**One-Command Solution**:
```bash
./quick_fix.sh
```

This script automatically:
- ✅ Cleans existing setup
- ✅ Fixes all Python 3.13 compatibility issues
- ✅ Sets up both backend and frontend
- ✅ Creates sample data
- ✅ Tests the installation
- ✅ Creates startup scripts

**Then start the app:**
```bash
./start_app.sh
```

---

## Common Setup Issues and Solutions

### 1. Python Version Compatibility Issues

**Problem**: Pandas/NumPy/lxml compilation errors with Python 3.13
```
error: too few arguments to function call, expected 6, have 5
error: call to undeclared function
```

**Solutions**:

#### Option A: Use Python 3.11 or 3.12 (Recommended)
```bash
# Install Python 3.11 or 3.12 via Homebrew (macOS)
brew install python@3.11
# or
brew install python@3.12

# Create virtual environment with specific Python version
python3.11 -m venv backend/venv
# or
python3.12 -m venv backend/venv
```

#### Option B: Use Alternative Requirements (Python 3.13)
The project now automatically detects Python 3.13 and uses compatible package versions.

#### Option C: Use Conda Environment
```bash
# Install Miniconda/Anaconda first, then:
conda create -n real-estate python=3.11
conda activate real-estate
cd backend
pip install -r requirements.txt
```

### 2. Package Installation Issues

**Problem**: `pip install` fails for specific packages

**Solutions**:

#### Update pip and setuptools
```bash
cd backend
source venv/bin/activate
pip install --upgrade pip setuptools wheel
```

#### Install problematic packages separately
```bash
# For macOS with Apple Silicon
pip install --no-binary=pandas pandas
pip install --no-binary=numpy numpy

# Or try pre-compiled wheels
pip install --only-binary=all pandas numpy scikit-learn
```

#### Use conda for scientific packages
```bash
conda install pandas numpy scikit-learn matplotlib seaborn
pip install fastapi uvicorn  # Install remaining packages with pip
```

### 3. Node.js/NPM Issues

**Problem**: Frontend dependencies fail to install

**Solutions**:

#### Update Node.js
```bash
# Install latest LTS version
# Via nvm (recommended)
nvm install --lts
nvm use --lts

# Via Homebrew (macOS)
brew install node
```

#### Clear npm cache
```bash
cd frontend
npm cache clean --force
rm -rf node_modules package-lock.json
npm install
```

#### Use Yarn instead of npm
```bash
npm install -g yarn
cd frontend
yarn install
```

### 4. Map Component Issues

**Problem**: Map doesn't load or shows blank screen

**Solutions**:

#### Check browser console
Open browser dev tools (F12) and look for JavaScript errors

#### Leaflet CSS issues
Ensure you have a stable internet connection for CDN resources, or install leaflet locally:
```bash
cd frontend
npm install leaflet @types/leaflet react-leaflet
```

### 5. API Connection Issues

**Problem**: Frontend can't connect to backend API

**Solutions**:

#### Check backend is running
```bash
curl http://localhost:8000/health
```

#### CORS issues
The backend is configured for `localhost:3000`. If using different ports, update `backend/api/main.py`:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:YOUR_PORT"],
    # ...
)
```

#### Environment variables
Check `frontend/.env.local`:
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### 6. Machine Learning Model Issues

**Problem**: Model training fails or predictions are poor

**Solutions**:

#### Memory issues
```bash
# Reduce sample data size in train_model.py
n_samples = 500  # Instead of 1000
```

#### Missing model files
```bash
# Regenerate model
cd backend
source venv/bin/activate
python ml_model/train_model.py
```

### 7. Database/Data Issues

**Problem**: No property data available

**Solutions**:

#### Generate sample data
```bash
./generate_sample_data.sh
```

#### Check data directory
```bash
ls -la data/
# Should contain raw_properties.csv
```

### 8. Performance Issues

**Problem**: Application runs slowly

**Solutions**:

#### Reduce data size
Edit `backend/ml_model/train_model.py`:
```python
# Limit properties processed
all_listings[:100]  # Instead of [:500]
```

#### Use faster models
In `train_model.py`, remove slower models:
```python
models = {
    'Random Forest': RandomForestRegressor(n_estimators=50, n_jobs=-1),
    # Remove XGBoost and Gradient Boosting for faster training
}
```

## Platform-Specific Issues

### macOS
- Install Xcode command line tools: `xcode-select --install`
- For Apple Silicon Macs, some packages may need specific versions

### Windows
- Install Microsoft Visual C++ Build Tools
- Consider using WSL2 for better compatibility

### Linux
- Install build essentials: `sudo apt-get install build-essential`
- Install Python development headers: `sudo apt-get install python3-dev`

## Getting Help

If you're still having issues:

1. **Check the logs**: Look at terminal output for specific error messages
2. **Search for the specific error**: Copy the exact error message and search online
3. **Create a minimal test**: Try running individual components separately
4. **Check versions**: Run `python --version`, `node --version`, etc.

## Quick Reset

If all else fails, clean reset:

```bash
# Remove all generated files
rm -rf backend/venv
rm -rf frontend/node_modules
rm -rf frontend/.next
rm -rf data/
rm -rf models/

# Re-run setup
./setup.sh
``` 