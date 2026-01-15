import os
import json
from flask import Flask, jsonify, request
import warnings
from dotenv import load_dotenv
import stripe
from datetime import datetime, timedelta
from flask_cors import CORS

# Firebase Admin SDK - needed for Stripe webhooks to update user subscriptions
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    FIREBASE_ADMIN_AVAILABLE = True
except ImportError:
    print("⚠️ firebase_admin not available - Stripe webhooks will not be able to update user subscriptions")
    FIREBASE_ADMIN_AVAILABLE = False
    firebase_admin = None
    credentials = None
    firestore = None

# Initialize the Flask app
app = Flask(__name__,
            static_url_path='',
            static_folder='public')

# Enable CORS for all routes (allows Flutter app to make requests)
CORS(app)

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Initialize Firebase Admin SDK
db = None
if FIREBASE_ADMIN_AVAILABLE:
    try:
        firebase_admin.get_app()
        print("Firebase app already initialized")
    except ValueError:
        try:
            # Try to get Firebase credentials from environment variable first
            firebase_creds_json = os.getenv('FIREBASE_CREDENTIALS_JSON')
            if firebase_creds_json:
                print("Using Firebase credentials from environment variable")
                cred_dict = json.loads(firebase_creds_json)
                cred = credentials.Certificate(cred_dict)
                firebase_admin.initialize_app(cred)
            elif os.path.exists('firebase-credentials.json'):
                print("Using firebase-credentials.json file")
                cred = credentials.Certificate('firebase-credentials.json')
                firebase_admin.initialize_app(cred)
            else:
                print("⚠️ No Firebase credentials found (neither FIREBASE_CREDENTIALS_JSON env var nor firebase-credentials.json file)")
                print("   Stripe webhooks will not be able to update user subscriptions")
                # Don't initialize Firebase if no credentials
                firebase_admin = None
        except Exception as e:
            print(f"Firebase initialization error: {e}")
            print("   Stripe webhooks will not be able to update user subscriptions")
            db = None
        else:
            if firebase_admin:
                try:
                    db = firestore.client()
                    print("Firebase Firestore client initialized successfully")
                except Exception as e:
                    print(f"Firestore client error: {e}")
                    db = None
else:
    print("⚠️ Firebase Admin SDK not available - Stripe webhooks will not update user subscriptions")

# Load environment variables
load_dotenv()

# Initialize Stripe
stripe.api_key = os.getenv('STRIPE_SECRET_KEY', '')
if stripe.api_key:
    print("✅ Stripe initialized successfully")
else:
    print("⚠️ Stripe secret key not found. Payment features will be disabled.")

# Initialize Stripe checkout module with Firestore client
stripe_module = None
try:
    import sys
    import importlib.util
    stripe_checkout_path = os.path.join(os.path.dirname(__file__), "stripe_checkout.py")
    if os.path.exists(stripe_checkout_path):
        spec = importlib.util.spec_from_file_location("stripe_checkout_module", stripe_checkout_path)
        stripe_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(stripe_module)
        if db is not None:
            stripe_module.set_firestore_client(db)
            print("✅ Stripe checkout module initialized with Firestore client")
        else:
            print("⚠️ Stripe checkout module loaded but Firestore not available")
    else:
        print("⚠️ stripe_checkout.py not found")
except Exception as e:
    print(f"⚠️ Error initializing stripe checkout module: {e}")

# Create directories
os.makedirs('uploads', exist_ok=True)
os.makedirs('snout_data', exist_ok=True)

# ============================================================================
# HEALTH CHECK AND ROOT ROUTE
# ============================================================================

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'service': 'Snout Scout Backend',
        'endpoints': {
            'stripe': {
                'create_checkout': '/create-checkout-session (POST)',
                'create_portal': '/create-portal-session (POST)',
                'webhook': '/stripe-webhook (POST)',
                'payment_intent': '/create-payment-intent (POST)',
            },
            'subscription': '/subscription/<user_id> (GET)',
            'static': {
                'checkout': '/checkout.html',
                'success': '/success.html',
                'cancel': '/cancel.html',
            }
        },
        'firebase': 'available' if db is not None else 'not initialized',
        'stripe': 'configured' if stripe.api_key else 'not configured'
    }), 200

@app.route('/health', methods=['GET'])
def health():
    """Simple health check"""
    return jsonify({'status': 'ok'}), 200

# ============================================================================
# STRIPE PAYMENT ENDPOINTS
# ============================================================================

@app.route('/create-payment-intent', methods=['POST'])
def create_payment_intent():
    """Create a Stripe payment intent for PRO subscription"""
    try:
        if not stripe.api_key:
            return jsonify({"error": "Stripe not configured"}), 500
        
        data = request.get_json()
        user_id = data.get('user_id')
        user_email = data.get('user_email')
        amount = data.get('amount', 300)  # Default £3.00 in pence
        currency = data.get('currency', 'gbp')
        
        if not user_id:
            return jsonify({"error": "user_id is required"}), 400
        
        print(f"💳 Creating payment intent for user: {user_id}, amount: {amount} {currency}")
        
        # Create payment intent
        payment_intent = stripe.PaymentIntent.create(
            amount=amount,
            currency=currency,
            metadata={
                'subscription_type': 'pro_monthly',
                'user_id': user_id,
                'user_email': user_email or '',
            },
            automatic_payment_methods={
                'enabled': True,
            },
        )
        
        print(f"✅ Payment intent created: {payment_intent.id}")
        
        return jsonify({
            'client_secret': payment_intent.client_secret,
            'amount': payment_intent.amount,
            'currency': payment_intent.currency,
            'payment_intent_id': payment_intent.id,
        }), 200
        
    except stripe.error.StripeError as e:
        print(f"❌ Stripe error: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        print(f"❌ Error creating payment intent: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route('/webhook', methods=['POST'])
def stripe_webhook():
    """Handle Stripe webhook events"""
    try:
        if not stripe.api_key:
            return jsonify({"error": "Stripe not configured"}), 500
        
        payload = request.data
        sig_header = request.headers.get('Stripe-Signature')
        webhook_secret = os.getenv('STRIPE_WEBHOOK_SECRET', '')
        
        if not webhook_secret:
            print("⚠️ STRIPE_WEBHOOK_SECRET not set, skipping signature verification")
            event = stripe.Event.construct_from(
                json.loads(payload), stripe.api_key
            )
        else:
            try:
                event = stripe.Webhook.construct_event(
                    payload, sig_header, webhook_secret
                )
            except ValueError as e:
                print(f"❌ Invalid payload: {e}")
                return jsonify({"error": "Invalid payload"}), 400
            except stripe.error.SignatureVerificationError as e:
                print(f"❌ Invalid signature: {e}")
                return jsonify({"error": "Invalid signature"}), 400
        
        # Handle the event
        event_type = event['type']
        print(f"📬 Received webhook event: {event_type}")
        
        if event_type == 'payment_intent.succeeded':
            payment_intent = event['data']['object']
            user_id = payment_intent['metadata'].get('user_id')
            
            if user_id:
                print(f"✅ Payment succeeded for user: {user_id}")
                update_user_subscription(user_id, payment_intent['id'], 'active')
            else:
                print("⚠️ No user_id in payment intent metadata")
                
        elif event_type == 'payment_intent.payment_failed':
            payment_intent = event['data']['object']
            user_id = payment_intent['metadata'].get('user_id')
            print(f"❌ Payment failed for user: {user_id}")
            
        elif event_type == 'payment_intent.canceled':
            payment_intent = event['data']['object']
            user_id = payment_intent['metadata'].get('user_id')
            print(f"⚠️ Payment canceled for user: {user_id}")
            
        else:
            print(f"ℹ️ Unhandled event type: {event_type}")
        
        return jsonify({"received": True}), 200
        
    except Exception as e:
        print(f"❌ Error handling webhook: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def update_user_subscription(user_id, payment_intent_id, status='active'):
    """Update user's subscription status in Firestore"""
    try:
        if db is None:
            print("⚠️ Firestore not initialized, cannot update subscription")
            return False
        
        # Calculate expiration date (1 month from now)
        expires_at = datetime.now() + timedelta(days=30)
        
        # Update user document
        user_ref = db.collection('Users').document(user_id)
        user_ref.update({
            'accountType': 'pro' if status == 'active' else 'basic',
            'subscription': {
                'is_pro': status == 'active',
                'status': status,
                'payment_intent_id': payment_intent_id,
                'expires_at': expires_at.isoformat(),
                'updated_at': datetime.now().isoformat(),
            },
            'kennel_capacity': 5 if status == 'active' else 3,
        })
        
        print(f"✅ Updated subscription for user {user_id}: {status}")
        return True
        
    except Exception as e:
        print(f"❌ Error updating subscription: {e}")
        import traceback
        traceback.print_exc()
        return False


@app.route('/subscription/<user_id>', methods=['GET'])
def get_subscription(user_id):
    """Get user's subscription status"""
    try:
        if db is None:
            return jsonify({
                'is_pro': False,
                'status': 'inactive',
                'error': 'Firestore not initialized'
            }), 500
        
        user_ref = db.collection('Users').document(user_id)
        user_doc = user_ref.get()
        
        if not user_doc.exists:
            return jsonify({
                'is_pro': False,
                'status': 'inactive',
                'error': 'User not found'
            }), 404
        
        user_data = user_doc.to_dict()
        subscription = user_data.get('subscription', {})
        
        # Check if subscription is expired
        is_pro = subscription.get('is_pro', False)
        expires_at_str = subscription.get('expires_at')
        
        if expires_at_str:
            expires_at = datetime.fromisoformat(expires_at_str)
            if datetime.now() > expires_at:
                is_pro = False
                user_ref.update({
                    'subscription.is_pro': False,
                    'subscription.status': 'expired',
                })
        
        return jsonify({
            'is_pro': is_pro,
            'status': subscription.get('status', 'inactive'),
            'expires_at': expires_at_str,
            'payment_intent_id': subscription.get('payment_intent_id'),
        }), 200
        
    except Exception as e:
        print(f"❌ Error getting subscription: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================================
# STRIPE CHECKOUT SESSION ROUTES
# ============================================================================

@app.route('/checkout.html', methods=['GET'])
def get_checkout():
    """Serve checkout page"""
    return app.send_static_file('checkout.html')

@app.route('/success.html', methods=['GET'])
def get_success():
    """Serve success page"""
    return app.send_static_file('success.html')

@app.route('/cancel.html', methods=['GET'])
def get_cancel():
    """Serve cancel page"""
    return app.send_static_file('cancel.html')

@app.route('/create-checkout-session', methods=['POST'])
def create_checkout_session_route():
    """Create Stripe Checkout Session"""
    try:
        if stripe_module is None:
            return jsonify({"error": "Stripe module not initialized"}), 500
        
        # Set request context for stripe_checkout module
        if stripe_module and hasattr(stripe_module, 'set_request_context'):
            stripe_module.set_request_context(request)
        
        lookup_key = request.form.get('lookup_key') or (request.json.get('lookup_key') if request.is_json else 'pro_monthly')
        return_json = request.is_json
        return stripe_module.create_checkout_session(lookup_key, return_json=return_json)
    except Exception as e:
        print(f"❌ Error in create_checkout_session route: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/create-portal-session', methods=['POST'])
def create_portal_session_route():
    """Create Stripe Customer Portal Session"""
    try:
        if not stripe.api_key:
            return jsonify({"error": "Stripe not configured"}), 500
        
        # Get data from request (can be form or JSON)
        if request.is_json:
            data = request.json
            customer_id = data.get('customer_id')
        else:
            customer_id = request.form.get('customer_id')
            session_id = request.form.get('session_id')
            if session_id and not customer_id:
                checkout_session = stripe.checkout.Session.retrieve(session_id)
                customer_id = checkout_session.customer
        
        if not customer_id:
            return jsonify({"error": "customer_id is required"}), 400
        
        return_url = request.json.get('return_url') if request.is_json else request.form.get('return_url')
        if not return_url:
            return_url = os.getenv('STRIPE_DOMAIN', 'http://localhost:5001')
        
        # Create portal session
        portal_session = stripe.billing_portal.Session.create(
            customer=customer_id,
            return_url=return_url,
        )
        
        print(f"✅ Portal session created: {portal_session.id}")
        return jsonify({'url': portal_session.url}), 200
        
    except stripe.error.StripeError as e:
        print(f"❌ Stripe error creating portal session: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        print(f"❌ Error creating portal session: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/stripe-webhook', methods=['POST'])
def stripe_webhook_route():
    """Handle Stripe webhook events"""
    try:
        if stripe_module is None:
            return jsonify({"error": "Stripe module not initialized"}), 500
        webhook_secret = os.getenv('STRIPE_WEBHOOK_SECRET', '')
        signature = request.headers.get('stripe-signature')
        payload = request.data
        
        return stripe_module.handle_webhook(payload, signature, webhook_secret)
    except Exception as e:
        print(f"❌ Error in stripe_webhook route: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
