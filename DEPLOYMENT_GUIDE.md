# Deployment Guide: Deploying ScoutSnout Backend to Render

This guide will help you deploy your Flask backend (with Stripe and ML features) to Render.com so your Flutter app can access it from anywhere.

## 🎯 Why Render?

- **Free tier available** (with limitations)
- **Easy Flask deployment**
- **Environment variable management**
- **Auto-deploys from GitHub**
- **HTTPS included**

## 📋 Prerequisites

1. A GitHub account
2. Your backend code pushed to GitHub
3. Your Stripe API keys
4. Your Firebase service account credentials

## 🚀 Step 1: Push Your Backend to GitHub

If you haven't already, push your `backend` folder to GitHub:

```bash
cd /Users/carlharicombe/IdeaProjects/ScoutSnout
git init  # if not already a git repo
git add backend/
git commit -m "Add backend code for deployment"
git remote add origin YOUR_GITHUB_REPO_URL
git push -u origin main
```

## 🔧 Step 2: Create a Render Account

1. Go to [render.com](https://render.com)
2. Sign up for a free account (or sign in with GitHub)

## 🌐 Step 3: Deploy the Backend

1. **Click "New +" → "Web Service"**
2. **Connect your GitHub repository** (authorize Render to access your repos)
3. **Select your repository** and the branch (usually `main`)
4. **Configure the service:**
   - **Name**: `scoutsnout-backend` (or any name you like)
   - **Region**: Choose closest to your users (e.g., `Oregon (US West)`)
   - **Branch**: `main` (or your default branch)
   - **Root Directory**: `backend` (important!)
   - **Runtime**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: Leave empty (Procfile will handle this)

## 🔑 Step 4: Set Environment Variables

In Render dashboard, go to your service → **Environment** tab, and add:

### Required Environment Variables:

```
STRIPE_SECRET_KEY=sk_test_your_stripe_secret_key_here
STRIPE_PUBLISHABLE_KEY=pk_test_your_stripe_publishable_key_here
STRIPE_WEBHOOK_SECRET=whsec_your_webhook_secret_here
STRIPE_DOMAIN=https://your-render-app.onrender.com

FIREBASE_CREDENTIALS={"type":"service_account",...your full JSON here...}

PORT=10000
```

### How to Get These Values:

#### Stripe Keys:
1. Go to [Stripe Dashboard](https://dashboard.stripe.com/test/apikeys)
2. Copy your **Secret Key** and **Publishable Key**

#### Stripe Webhook Secret:
1. In Stripe Dashboard → **Developers** → **Webhooks**
2. Create a webhook endpoint: `https://your-render-app.onrender.com/stripe-webhook`
3. Copy the **Signing Secret** (starts with `whsec_`)

#### Firebase Credentials:
1. In Firebase Console → **Project Settings** → **Service Accounts**
2. Click **Generate New Private Key**
3. Download the JSON file
4. Copy the entire JSON content and paste it as the `FIREBASE_CREDENTIALS` value (keep it as one line JSON)

#### STRIPE_DOMAIN:
- After deployment, Render will give you a URL like: `https://scoutsnout-backend.onrender.com`
- Use this as your `STRIPE_DOMAIN`

## 📝 Step 5: Update app.py for Production

The app.py should already handle environment variables. Make sure it:
- Uses `os.getenv()` for sensitive values
- Uses `PORT` environment variable (Render sets this automatically)
- Handles CORS if needed

## 🚦 Step 6: Update Flutter App URLs

After deployment, you'll get a URL like: `https://scoutsnout-backend-xyz.onrender.com`

Update all backend URLs in your Flutter app:

### Option A: Create a Config File (Recommended)

Create `lib/config/app_config.dart`:

```dart
class AppConfig {
  // Update this after deployment
  static const String backendUrl = 'https://your-app-name.onrender.com';
  
  // For development (local network)
  // static const String backendUrl = 'http://192.168.0.28:5001';
}
```

### Option B: Update Each File Manually

Update these files to use your Render URL:

1. `lib/Services/stripe_service.dart` - line 14
2. `lib/Pages/scan_lost_pet.dart` - line 19
3. `lib/Components/lost_pet_scanner.dart` - line 101
4. `lib/Components/snout_scanner.dart` - line 126
5. `lib/Pages/create_my_pet.dart` - lines 300, 750

## ⚠️ Important Notes

### File Size Limitations:
- **Render Free Tier**: 500MB total
- Your ML models might be large (TensorFlow, PyTorch, etc.)
- Consider:
  - Using Render's paid tier for larger storage
  - Hosting models on cloud storage (S3, GCS) and loading at runtime
  - Using a lighter model

### Cold Starts:
- Free tier services "sleep" after 15 minutes of inactivity
- First request after sleep takes ~30 seconds (cold start)
- Paid tier doesn't sleep

### Timeout:
- Default timeout is 30 seconds
- ML processing might take longer
- Consider using Render's background workers for heavy ML tasks

### CORS (Cross-Origin Resource Sharing):
If you get CORS errors, add this to your Flask app:

```python
from flask_cors import CORS
app = Flask(__name__)
CORS(app)  # Allow all origins (adjust for production)
```

Add `flask-cors>=3.0.10` to `requirements.txt`

## 🔍 Step 7: Test Your Deployment

1. Once deployed, visit: `https://your-app-name.onrender.com/`
2. Test Stripe endpoints
3. Test ML endpoints
4. Check Render logs for errors

## 📊 Monitoring

- **Logs**: Render dashboard → Your service → **Logs** tab
- **Metrics**: Monitor CPU, memory usage
- **Alerts**: Set up alerts for errors

## 🔄 Updating Your Backend

1. Push changes to GitHub
2. Render automatically redeploys (or manually trigger in dashboard)
3. Wait ~5 minutes for deployment

## 🆘 Troubleshooting

### "Build failed"
- Check `requirements.txt` for missing dependencies
- Check Python version compatibility
- View build logs in Render dashboard

### "Service won't start"
- Check `Procfile` format
- Verify environment variables are set
- Check startup logs

### "Timeout errors"
- Increase timeout in `Procfile`: `--timeout 600`
- Consider using background workers for heavy tasks

### "CORS errors"
- Add `flask-cors` to requirements
- Configure CORS in Flask app

## 💰 Cost Considerations

- **Free Tier**: Good for development/testing
- **Starter Plan ($7/month)**: No sleep, more resources
- **Professional Plan ($25/month)**: Better for production with ML models

---

## 🎉 After Deployment

Once deployed, update your Flutter app's backend URL and you're ready to go!

Your backend will be accessible at: `https://your-app-name.onrender.com`

All your API endpoints will work:
- `/scanFace` - Snout scanning
- `/identifyPet` - Lost pet identification
- `/create-checkout-session` - Stripe checkout
- `/create-customer-portal-session` - Stripe portal
- `/stripe-webhook` - Stripe webhooks
- And all other endpoints!
