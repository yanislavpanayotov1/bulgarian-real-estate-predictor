# 🏠 Bulgarian Real Estate Price Predictor

**AI-powered web application for predicting real estate prices in Bulgaria using machine learning**

[![Next.js](https://img.shields.io/badge/Next.js-14.0-black?style=flat-square&logo=next.js)](https://nextjs.org/)
[![Python](https://img.shields.io/badge/Python-3.13-blue?style=flat-square&logo=python)](https://python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--learn%20%7C%20XGBoost-orange?style=flat-square)](https://scikit-learn.org/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0-blue?style=flat-square&logo=typescript)](https://typescriptlang.org/)

> A full-stack machine learning application that predicts Bulgarian property prices with 68% accuracy using multiple AI algorithms, featuring an interactive map interface and real-time market analytics.

## 🌟 Features

### 🤖 **Machine Learning Engine**
- **5 ML Models**: Random Forest, XGBoost, Gradient Boosting, Ridge Regression, Linear Regression
- **Smart Feature Engineering**: Price per sqm, room density, property age, location encoding
- **Real-time Predictions**: Instant price estimates based on property characteristics
- **Model Comparison**: Automatic selection of best-performing algorithm

### 🗺️ **Interactive Interface**
- **Dynamic Map**: Leaflet.js integration with Bulgarian coordinates
- **Property Visualization**: Color-coded markers based on price ranges
- **City Navigation**: Quick jump to Sofia, Plovdiv, Varna, Burgas, Stara Zagora
- **Responsive Design**: Optimized for desktop and mobile devices

### 📊 **Analytics Dashboard**
- **Market Statistics**: Average prices, trends, and city comparisons
- **Property Filtering**: Search by city, price range, size, and features
- **Real-time Data**: Live market insights and neighborhood analysis
- **Performance Metrics**: Model accuracy and confidence intervals

### 🏗️ **Modern Architecture**
- **Frontend**: Next.js 14, React 18, TypeScript, TailwindCSS
- **Backend**: FastAPI, Python 3.13, Async/Await architecture
- **Database**: CSV-based with pandas processing
- **APIs**: RESTful endpoints with OpenAPI documentation

## 🚀 Quick Start

### Prerequisites
- **Python 3.11+** (Python 3.13 recommended)
- **Node.js 18+**
- **npm** or **yarn**

### One-Command Setup
```bash
./start_app.sh
```

### Manual Setup
```bash
# Backend Setup
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Frontend Setup
cd ../frontend
npm install

# Start Development Servers
cd ../backend && uvicorn api.main:app --reload &
cd ../frontend && npm run dev
```

### 🌐 Access Points
- **Frontend Application**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## 🛠️ Technology Stack

### **Backend**
```python
FastAPI          # Modern, fast web framework
Scikit-learn     # Machine learning algorithms
XGBoost          # Gradient boosting framework
Pandas           # Data manipulation and analysis
NumPy            # Numerical computing
Uvicorn          # ASGI server implementation
Pydantic         # Data validation and settings
Geopy            # Geocoding and location services
```

### **Frontend**
```javascript
Next.js 14       // React framework with SSR
TypeScript       // Type-safe JavaScript
TailwindCSS      // Utility-first CSS framework
React Leaflet    // Interactive maps
Chart.js         // Data visualization
React Query      // Data fetching and caching
React Hook Form  // Form handling
Zod              // Schema validation
```

### **Machine Learning Pipeline**
1. **Data Loading**: CSV parsing with pandas
2. **Feature Engineering**: Location encoding, price per sqm calculation
3. **Model Training**: Multiple algorithm comparison
4. **Hyperparameter Tuning**: Grid search optimization
5. **Model Selection**: Best performer based on R² and MAE
6. **Prediction API**: Real-time inference endpoint

## 📊 Model Performance

| Algorithm | MAE (BGN) | R² Score | Training Time |
|-----------|-----------|----------|---------------|
| **Ridge Regression** | 23,402 | **0.685** | 0.12s |
| Linear Regression | 23,406 | 0.685 | 0.08s |
| Gradient Boosting | 23,849 | 0.679 | 2.34s |
| Random Forest | 24,097 | 0.652 | 1.67s |
| XGBoost | 25,622 | 0.594 | 0.89s |

*Best model automatically selected based on validation performance*

## 🌍 Deployment

### **Free Deployment (Recommended)**
```bash
# Deploy Backend to Railway
./deploy_backend.sh

# Deploy Frontend to Vercel
./deploy_frontend.sh
```

### **Deployment Platforms**
- **Frontend**: Vercel (unlimited free projects)
- **Backend**: Railway ($5/month free credit)
- **Total Cost**: $0/month within free tiers

See [`DEPLOYMENT_GUIDE.md`](./DEPLOYMENT_GUIDE.md) for detailed instructions.

## 📁 Project Structure

```
Real Estate Price predictor/
├── backend/                 # FastAPI backend
│   ├── api/                # API endpoints
│   │   └── main.py        # Main FastAPI application
│   ├── ml_model/          # Machine learning pipeline
│   │   └── train_model.py # ML model training and prediction
│   ├── scraper/           # Data scraping utilities
│   │   └── imoti_scraper.py # Property data scraper
│   └── requirements.txt   # Python dependencies
├── frontend/              # Next.js frontend
│   ├── pages/            # Next.js pages
│   │   ├── index.tsx     # Main application page
│   │   └── _app.tsx      # App configuration
│   ├── components/       # React components
│   │   └── PropertyMap.tsx # Interactive map component
│   └── package.json      # Node.js dependencies
├── data/                 # Data storage
│   └── raw_properties.csv # Property dataset (999 records)
├── models/               # Trained ML models
│   └── best_model.joblib # Serialized best model
├── deploy_backend.sh     # Railway deployment script
├── deploy_frontend.sh    # Vercel deployment script
└── start_app.sh         # Development server startup
```

## 🔧 API Endpoints

### **Prediction**
```http
POST /predict
Content-Type: application/json

{
  "size": 75,
  "rooms": 2,
  "floor": 3,
  "year_built": 2010,
  "city": "Sofia",
  "neighborhood": "Center",
  "features": ["parking", "balcony"]
}
```

### **Properties**
```http
GET /properties?city=Sofia&min_price=50000&max_price=200000&limit=50
```

### **Market Statistics**
```http
GET /market-stats
```

### **Neighborhoods**
```http
GET /neighborhoods
```

See full API documentation at `/docs` when running the server.

## 📈 Sample Data

The application includes **999 realistic Bulgarian property records** with:
- **Price Range**: 20,000 - 340,000 BGN
- **Size Range**: 25 - 180 m²
- **Cities**: Sofia, Plovdiv, Varna, Burgas, Stara Zagora
- **Features**: Parking, balcony, elevator
- **Coordinates**: Accurate Bulgarian geographic locations

## 🧪 Testing

```bash
# Backend Tests
cd backend
python test_setup.py

# API Health Check
curl http://localhost:8000/health

# Sample Prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"size": 75, "rooms": 2, "city": "Sofia"}'
```

## 🚧 Troubleshooting

### **Common Issues**

#### Python 3.13 Compatibility
```bash
# Use the quick fix script
./quick_fix.sh
```

#### Frontend Build Errors
```bash
cd frontend
rm -rf .next node_modules
npm install
npm run dev
```

#### Model Loading Issues
```bash
cd backend
python -c "from ml_model.train_model import RealEstatePricePredictor; RealEstatePricePredictor().load_model('../models')"
```

See [`TROUBLESHOOTING.md`](./TROUBLESHOOTING.md) for detailed solutions.

## 🤝 Contributing

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **FastAPI** for the amazing web framework
- **Scikit-learn** for robust ML algorithms
- **Next.js** for the powerful React framework
- **OpenStreetMap** for map data via Leaflet
- **Bulgarian Real Estate Market** for inspiration

## 📬 Contact

**Yanislav** - [GitHub Profile](https://github.com/yourusername)

**Project Link**: https://github.com/yourusername/bulgarian-real-estate-predictor

---

⭐ **Star this repository if you found it helpful!**

*Built with ❤️ for the Bulgarian real estate market* 