# Render Deployment - Python Version Issue

## Problem
Render is using Python 3.13 instead of Python 3.11, causing build failures.

## Solution: Set Python Version in Render Dashboard

### Option 1: Via Dashboard (RECOMMENDED)
1. Go to: https://dashboard.render.com
2. Click on your `snout-scan` service
3. Go to **Settings** tab
4. Scroll to **"Build & Deploy"** section
5. Find **"Python Version"** field
6. **Change it from "auto" or "3.13" to `3.11.0`**
7. Click **Save Changes**
8. Trigger a new deploy

### Option 2: Via Render CLI
If you have Render CLI installed:
```bash
render services:update snout-scan --python-version 3.11.0
```

### Option 3: Delete and Recreate Service
1. Delete the current service in Render dashboard
2. Create a new Web Service
3. **During creation, explicitly select Python 3.11**
4. Point to the same GitHub repo

## Why This is Happening
- Python 3.13 is very new (released Oct 2024)
- Many packages don't have pre-built wheels for 3.13 yet
- They try to build from source and fail
- Python 3.11 has better wheel support

## After Setting Python Version
Once Python 3.11 is set, the minimal requirements should install successfully.
