#!/bin/bash
# Script to download ML models required for the backend
# Run this after deployment or during setup

echo "📥 Downloading ML models for ScoutSnout backend..."

# Create models directory if it doesn't exist
mkdir -p models/ssd_mobilenet_v2_coco_2018_03_29

# Download SSD MobileNet V2 COCO model (if not using a direct download link)
# You'll need to provide your own download links or host these on cloud storage
echo "⚠️  Model files need to be downloaded manually or from cloud storage"
echo ""
echo "For deployment on Render/Heroku:"
echo "1. Upload models to cloud storage (Google Cloud Storage, AWS S3, etc.)"
echo "2. Download them during build or at runtime"
echo "3. Or use Git LFS (see DEPLOYMENT_GUIDE.md)"
echo ""
echo "Alternatively, you can download models directly during app startup"
echo "in app.py if you host them on a cloud storage service."
