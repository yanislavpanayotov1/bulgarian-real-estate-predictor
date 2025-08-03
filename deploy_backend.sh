#!/bin/bash

echo "🚀 Deploying Backend to Railway..."
echo "================================="

# Install Railway CLI if not installed
if ! command -v railway &> /dev/null; then
    echo "📦 Installing Railway CLI..."
    curl -fsSL https://railway.app/install.sh | sh
    export PATH="$PATH:/Users/$(whoami)/.railway/bin"
fi

# Go to backend directory
cd backend

# Create Procfile for Railway
echo "🔧 Creating deployment configuration..."
cat > Procfile << 'EOF'
web: uvicorn api.main:app --host 0.0.0.0 --port $PORT
EOF

# Create railway.toml
cat > railway.toml << 'EOF'
[build]
builder = "nixpacks"

[deploy]
startCommand = "uvicorn api.main:app --host 0.0.0.0 --port $PORT"
healthcheckPath = "/health"
healthcheckTimeout = 300
restartPolicyType = "ON_FAILURE"

[[services]]
name = "bulgarian-real-estate-api"
EOF

# Create .railwayignore
cat > .railwayignore << 'EOF'
venv/
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
.env
.venv/
pip-log.txt
pip-delete-this-directory.txt
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
*.log
.git/
.mypy_cache/
.pytest_cache/
.hypothesis/
.DS_Store
*.sqlite
*.db
EOF

# Create optimized requirements.txt for deployment
echo "📦 Creating optimized requirements for deployment..."
cat > requirements.txt << 'EOF'
fastapi==0.104.1
uvicorn[standard]==0.24.0
pandas==2.2.0
numpy==1.26.0
scikit-learn==1.4.0
xgboost==2.0.3
requests==2.31.0
beautifulsoup4==4.13.4
html5lib==1.1
geopy==2.4.1
joblib==1.3.2
python-multipart==0.0.6
pydantic==2.5.0
python-dotenv==1.0.0
aiofiles==23.2.1
httpx==0.25.2
EOF

# Ensure models directory exists and has content
if [ ! -d "../models" ] || [ ! -f "../models/best_model.joblib" ]; then
    echo "🤖 Training model for deployment..."
    venv/bin/python -c "
import sys
sys.path.append('.')
from ml_model.train_model import RealEstatePricePredictor

# Train and save model
predictor = RealEstatePricePredictor()
df = predictor.load_data('../data/raw_properties.csv')
if df.empty:
    df = predictor._create_sample_data()
    df.to_csv('../data/raw_properties.csv', index=False)

df = predictor.feature_engineering(df)
X, y = predictor.prepare_features(df)
predictor.train_models(X, y)
predictor.save_model('../models')
print('✅ Model trained and saved for deployment')
"
fi

# Copy models and data to backend directory for deployment
echo "📁 Copying models and data for deployment..."
cp -r ../models . 2>/dev/null || true
cp -r ../data . 2>/dev/null || true

# Update API paths for deployment
echo "🔧 Updating API paths for deployment..."
sed -i.bak 's|../data/raw_properties.csv|data/raw_properties.csv|g' api/main.py
sed -i.bak 's|../models|models|g' ml_model/train_model.py

echo ""
echo "✅ Backend prepared for Railway deployment!"
echo ""
echo "🎯 Next steps:"
echo "1. Run: railway login"
echo "2. Run: railway init"
echo "3. Run: railway up"
echo ""
echo "📝 After deployment, your API will be available at:"
echo "   https://your-project-name.railway.app"
echo ""
echo "🔗 Update your frontend with the Railway URL!"
echo "" 