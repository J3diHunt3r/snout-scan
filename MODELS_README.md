# ML Models - Setup Instructions

The ML model files are too large for GitHub (100MB+), so they're excluded from the repository.

## For Local Development

If you already have the models in your `backend/` directory, they'll work as-is. The `.gitignore` prevents them from being committed.

## For Deployment (Render/Heroku/etc.)

You have several options:

### Option 1: Download Models at Runtime (Recommended)

Modify `app.py` to download models from cloud storage (Google Cloud Storage, AWS S3, etc.) on first startup:

```python
import os
import requests
from pathlib import Path

def ensure_models_exist():
    model_dir = Path('models/ssd_mobilenet_v2_coco_2018_03_29')
    model_file = model_dir / 'frozen_inference_graph.pb'
    
    if not model_file.exists():
        print("📥 Downloading models from cloud storage...")
        # Download from your cloud storage URL
        # Example with Google Cloud Storage:
        url = os.getenv('MODEL_DOWNLOAD_URL')
        if url:
            response = requests.get(url)
            model_dir.mkdir(parents=True, exist_ok=True)
            with open(model_file, 'wb') as f:
                f.write(response.content)
        else:
            print("⚠️ Warning: Models not found and MODEL_DOWNLOAD_URL not set")

# Call this at app startup
ensure_models_exist()
```

### Option 2: Use Git LFS (Git Large File Storage)

If you want to track models in git:

1. Install Git LFS: `brew install git-lfs` (Mac) or see https://git-lfs.github.com
2. Initialize in your repo: `git lfs install`
3. Track large files: `git lfs track "*.pb" "*.tar.gz" "*.ckpt.*"`
4. Add and commit normally

**Note**: Git LFS requires a paid GitHub account for large files or use a free LFS host.

### Option 3: Host on Cloud Storage

1. Upload models to Google Cloud Storage / AWS S3 / Azure Blob
2. Set download URLs as environment variables
3. Download during deployment build or app startup

### Option 4: Store Models in Render's Persistent Disk (Paid Plans)

Render paid plans support persistent storage. You can:
- Upload models once manually via SSH
- Or download them during first deployment

## Required Model Files

- `models/ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb` (~66 MB)
- `models/ssd_mobilenet_v2_coco_2018_03_29/model.ckpt.*` (~64 MB each)
- `models/ssd_mobilenet_v2_coco_2018_03_29/saved_model/saved_model.pb` (~66 MB)
- `yolov5s.pt` (YOLOv5 model)

## For Now (Quick Fix)

If you need to deploy immediately without models, you can:
1. Deploy the backend without ML features first
2. Add model downloading later
3. Or use a lighter model that fits within size limits
