#!/bin/bash

# ScoutSnout Backend Server Startup Script
# This script checks dependencies and starts the Flask server

echo "🚀 Starting ScoutSnout Backend Server..."
echo "========================================"

# Check if we're in the backend directory
if [ ! -f "app.py" ]; then
    echo "❌ Error: app.py not found. Please run this script from the backend directory."
    exit 1
fi

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 not found. Please install Python 3."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment not found. Creating one..."
    python3 -m venv venv
    echo "✅ Virtual environment created."
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Check if requirements are installed
if ! python3 -c "import flask" 2>/dev/null; then
    echo "⚠️  Dependencies not installed. Installing requirements..."
    pip install -r requirements.txt
    echo "✅ Dependencies installed."
fi

# Get the local IP address
LOCAL_IP=$(ipconfig getifaddr en0 2>/dev/null || ipconfig getifaddr en1 2>/dev/null || echo "localhost")
echo ""
echo "🌐 Server will be accessible at:"
echo "   Local: http://localhost:5001"
echo "   Network: http://$LOCAL_IP:5001"
echo ""
echo "📱 Make sure your Flutter app is configured to use: http://$LOCAL_IP:5001"
echo ""
echo "🔄 Starting Flask server..."
echo "   (Press Ctrl+C to stop)"
echo ""

# Start the Flask server
python3 app.py












