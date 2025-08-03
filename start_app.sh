#!/bin/bash

echo "🏠 Starting Bulgarian Real Estate Price Predictor..."
echo "=================================================="
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Stopping services..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start backend
echo "📊 Starting backend API server..."
cd backend
venv/bin/python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
cd ..

# Wait for backend to start
echo "⏳ Waiting for backend to initialize..."
sleep 8

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
echo "   📱 Frontend:     http://localhost:3000"
echo "   🔌 Backend API:  http://localhost:8000"
echo "   📖 API Docs:     http://localhost:8000/docs"
echo ""
echo "🎯 Open http://localhost:3000 to use the app!"
echo ""
echo "Press Ctrl+C to stop both services"
echo ""

# Wait for processes
wait $BACKEND_PID $FRONTEND_PID 