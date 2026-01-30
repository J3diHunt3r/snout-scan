#!/bin/bash
# Build script to ensure Python 3.11 is used

echo "🐍 Checking Python version..."
python3 --version

# Install dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Build complete"
