"""
stripe.py
Stripe Checkout Session and Customer Portal handlers
"""
import os
import json
from flask import redirect, jsonify, request, send_from_directory
import stripe
from datetime import datetime, timedelta

# Initialize Stripe
stripe.api_key = os.getenv('STRIPE_SECRET_KEY', '')

# Domain configuration - update this for production
YOUR_DOMAIN = os.getenv('STRIPE_DOMAIN', 'http://localhost:5001')

# Get Firestore database instance - will be set from app.py
db = None

# Global request object - will be set from app.py
_request = None

def set_firestore_client(firestore_client):
    """Set the Firestore client from app.py"""
    global db
    db = firestore_client

def set_request_context(request_obj):
    """Set the request context from app.py"""
    global _request
    _request = request_obj


def create_checkout_session(lookup_key='pro_monthly', return_json=False):
    """Create a Stripe Checkout Session for subscription"""
    try:
        if not stripe.api_key:
            return jsonify({"error": "Stripe not configured"}), 500
        
        # Get the price for the lookup key
        prices = stripe.Price.list(
            lookup_keys=[lookup_key],
            expand=['data.product']
        )
        
        if not prices.data:
            return jsonify({"error": f"Price with lookup_key '{lookup_key}' not found"}), 404
        
        # Get user ID from request (if available)
        # Try global _request first, then fallback to Flask's request
        user_id = None
        try:
            req_obj = _request if _request is not None else request
            if req_obj:
                if hasattr(req_obj, 'is_json') and req_obj.is_json:
                    if hasattr(req_obj, 'json'):
                        user_id = req_obj.json.get('user_id')
                elif hasattr(req_obj, 'form'):
                    user_id = req_obj.form.get('user_id')
        except Exception as e:
            print(f"⚠️ Could not get user_id from request: {e}")
            pass
        
        checkout_session = stripe.checkout.Session.create(
            line_items=[
                {
                    'price': prices.data[0].id,
                    'quantity': 1,
                },
            ],
            mode='subscription',
            success_url=YOUR_DOMAIN + '/success.html?session_id={CHECKOUT_SESSION_ID}',
            cancel_url=YOUR_DOMAIN + '/cancel.html',
            metadata={
                'user_id': user_id or '',
                'subscription_type': lookup_key,  # Store which type of subscription (pro_monthly or business_monthly)
            },
            allow_promotion_codes=True,
        )
        
        print(f"✅ Checkout session created: {checkout_session.id}")
        
        # Return JSON if requested (for Flutter app), otherwise redirect (for web)
        if return_json:
            return jsonify({'url': checkout_session.url, 'session_id': checkout_session.id}), 200
        else:
            return redirect(checkout_session.url, code=303)
        
    except stripe.error.StripeError as e:
        print(f"❌ Stripe error creating checkout session: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        print(f"❌ Error creating checkout session: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def create_portal_session(session_id):
    """Create a Stripe Customer Portal session"""
    try:
        if not stripe.api_key:
            return jsonify({"error": "Stripe not configured"}), 500
        
        # Retrieve the checkout session to get the customer
        checkout_session = stripe.checkout.Session.retrieve(session_id)
        
        if not checkout_session.customer:
            return jsonify({"error": "No customer found in session"}), 400
        
        # Create portal session
        portal_session = stripe.billing_portal.Session.create(
            customer=checkout_session.customer,
            return_url=YOUR_DOMAIN,
        )
        
        print(f"✅ Portal session created: {portal_session.id}")
        return redirect(portal_session.url, code=303)
        
    except stripe.error.StripeError as e:
        print(f"❌ Stripe error creating portal session: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        print(f"❌ Error creating portal session: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def handle_webhook(payload, signature, webhook_secret):
    """Handle Stripe webhook events"""
    try:
        if not webhook_secret:
            print("⚠️ Webhook secret not configured, skipping signature verification")
            event = stripe.Event.construct_from(
                json.loads(payload), stripe.api_key
            )
        else:
            event = stripe.Webhook.construct_event(
                payload=payload,
                sig_header=signature,
                secret=webhook_secret
            )
        
        event_type = event['type']
        data = event['data']
        data_object = data['object']
        
        print(f"📬 Received webhook event: {event_type}")
        
        # Handle different event types
        if event_type == 'checkout.session.completed':
            print('🔔 Payment succeeded!')
            handle_checkout_completed(data_object)
            
        elif event_type == 'customer.subscription.created':
            print(f'✅ Subscription created: {event.id}')
            handle_subscription_created(data_object)
            
        elif event_type == 'customer.subscription.updated':
            print(f'🔄 Subscription updated: {event.id}')
            handle_subscription_updated(data_object)
            
        elif event_type == 'customer.subscription.deleted':
            print(f'❌ Subscription canceled: {event.id}')
            handle_subscription_deleted(data_object)
            
        elif event_type == 'customer.subscription.trial_will_end':
            print(f'⚠️ Subscription trial will end: {event.id}')
            
        elif event_type == 'invoice.payment_succeeded':
            print(f'💳 Invoice payment succeeded: {event.id}')
            handle_invoice_payment_succeeded(data_object)
            
        elif event_type == 'invoice.payment_failed':
            print(f'❌ Invoice payment failed: {event.id}')
            handle_invoice_payment_failed(data_object)
        
        return jsonify({'status': 'success'}), 200
        
    except ValueError as e:
        print(f"❌ Invalid payload: {e}")
        return jsonify({"error": "Invalid payload"}), 400
    except stripe.error.SignatureVerificationError as e:
        print(f"❌ Invalid signature: {e}")
        return jsonify({"error": "Invalid signature"}), 400
    except Exception as e:
        print(f"❌ Error handling webhook: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def handle_checkout_completed(session):
    """Handle checkout.session.completed event"""
    try:
        user_id = session.get('metadata', {}).get('user_id')
        customer_id = session.get('customer')
        subscription_id = session.get('subscription')
        
        if user_id and db:
            # Update user subscription in Firestore
            expires_at = datetime.now() + timedelta(days=30)
            
            user_ref = db.collection('Users').document(user_id)
            user_ref.update({
                'accountType': 'pro',  # Upgrade to PRO account
                'subscription': {
                    'is_pro': True,
                    'status': 'active',
                    'customer_id': customer_id,
                    'subscription_id': subscription_id,
                    'expires_at': expires_at.isoformat(),
                    'updated_at': datetime.now().isoformat(),
                },
                'kennel_capacity': 5,  # PRO gets 5 pets
            })
            
            print(f"✅ Updated subscription for user {user_id}")
    except Exception as e:
        print(f"❌ Error handling checkout completed: {e}")


def handle_subscription_created(subscription):
    """Handle customer.subscription.created event"""
    try:
        customer_id = subscription.get('customer')
        subscription_id = subscription.get('id')
        
        # Get customer to find user_id from metadata
        customer = stripe.Customer.retrieve(customer_id)
        user_id = customer.metadata.get('user_id')
        
        if user_id and db:
            expires_at = datetime.fromtimestamp(subscription.get('current_period_end', 0))
            
            # Get subscription to determine type from metadata or price
            subscription_obj = stripe.Subscription.retrieve(subscription_id)
            price_id = subscription_obj['items']['data'][0]['price']['id'] if subscription_obj.get('items', {}).get('data') else None
            
            # Try to determine account type from customer metadata or subscription
            subscription_type = 'pro_monthly'  # Default
            if customer.metadata.get('subscription_type'):
                subscription_type = customer.metadata.get('subscription_type')
            
            is_business = subscription_type == 'business_monthly'
            account_type = 'business' if is_business else 'pro'
            kennel_capacity = 10 if is_business else 5
            
            user_ref = db.collection('Users').document(user_id)
            update_data = {
                'accountType': account_type,
                'paymentPending': False,
                'pendingAccountType': None,
                'subscription': {
                    'is_pro': True,
                    'status': subscription.get('status', 'active'),
                    'customer_id': customer_id,
                    'subscription_id': subscription_id,
                    'expires_at': expires_at.isoformat(),
                    'updated_at': datetime.now().isoformat(),
                    'subscription_type': subscription_type,
                },
                'kennel_capacity': kennel_capacity,
            }
            
            if is_business:
                user_doc = user_ref.get()
                if user_doc.exists:
                    user_data = user_doc.to_dict()
                    update_data['isBusinessAccount'] = True
            
            user_ref.update(update_data)
            
            print(f"✅ Created subscription for user {user_id} as {account_type} account")
    except Exception as e:
        print(f"❌ Error handling subscription created: {e}")


def handle_subscription_updated(subscription):
    """Handle customer.subscription.updated event"""
    try:
        customer_id = subscription.get('customer')
        subscription_id = subscription.get('id')
        status = subscription.get('status')
        
        # Get customer to find user_id
        customer = stripe.Customer.retrieve(customer_id)
        user_id = customer.metadata.get('user_id')
        
        if user_id and db:
            expires_at = datetime.fromtimestamp(subscription.get('current_period_end', 0))
            is_active = status in ['active', 'trialing']
            
            # Get subscription type from existing user data
            user_ref = db.collection('Users').document(user_id)
            user_doc = user_ref.get()
            subscription_type = 'pro_monthly'  # Default
            if user_doc.exists:
                user_data = user_doc.to_dict()
                existing_subscription = user_data.get('subscription', {})
                subscription_type = existing_subscription.get('subscription_type', 'pro_monthly')
            
            is_business = subscription_type == 'business_monthly'
            account_type = 'business' if (is_business and is_active) else ('pro' if is_active else 'basic')
            kennel_capacity = 10 if (is_business and is_active) else (5 if is_active else 3)
            
            update_data = {
                'accountType': account_type,
                'subscription': {
                    'is_pro': is_active,
                    'status': status,
                    'customer_id': customer_id,
                    'subscription_id': subscription_id,
                    'expires_at': expires_at.isoformat(),
                    'updated_at': datetime.now().isoformat(),
                    'subscription_type': subscription_type,
                },
                'kennel_capacity': kennel_capacity,
            }
            
            # If it's a business account, ensure business flag is set correctly
            if is_business and is_active:
                if user_doc.exists:
                    user_data = user_doc.to_dict()
                    update_data['isBusinessAccount'] = True
            
            user_ref.update(update_data)
            
            print(f"✅ Updated subscription for user {user_id}: {status} ({account_type})")
    except Exception as e:
        print(f"❌ Error handling subscription updated: {e}")


def handle_subscription_deleted(subscription):
    """Handle customer.subscription.deleted event"""
    try:
        customer_id = subscription.get('customer')
        
        # Get customer to find user_id
        customer = stripe.Customer.retrieve(customer_id)
        user_id = customer.metadata.get('user_id')
        
        if user_id and db:
            user_ref = db.collection('Users').document(user_id)
            user_ref.update({
                'accountType': 'basic',  # Downgrade to basic account
                'subscription': {
                    'is_pro': False,
                    'status': 'canceled',
                    'updated_at': datetime.now().isoformat(),
                },
                'kennel_capacity': 3,  # Back to free tier
            })
            
            print(f"✅ Canceled subscription for user {user_id}")
    except Exception as e:
        print(f"❌ Error handling subscription deleted: {e}")


def handle_invoice_payment_succeeded(invoice):
    """Handle invoice.payment_succeeded event"""
    try:
        customer_id = invoice.get('customer')
        subscription_id = invoice.get('subscription')
        
        # Get customer to find user_id
        customer = stripe.Customer.retrieve(customer_id)
        user_id = customer.metadata.get('user_id')
        
        if user_id and db and subscription_id:
            # Get subscription to update expiry
            subscription = stripe.Subscription.retrieve(subscription_id)
            expires_at = datetime.fromtimestamp(subscription.get('current_period_end', 0))
            
            user_ref = db.collection('Users').document(user_id)
            user_ref.update({
                'subscription.expires_at': expires_at.isoformat(),
                'subscription.updated_at': datetime.now().isoformat(),
            })
            
            print(f"✅ Invoice payment succeeded for user {user_id}")
    except Exception as e:
        print(f"❌ Error handling invoice payment succeeded: {e}")


def handle_invoice_payment_failed(invoice):
    """Handle invoice.payment_failed event"""
    try:
        customer_id = invoice.get('customer')
        
        # Get customer to find user_id
        customer = stripe.Customer.retrieve(customer_id)
        user_id = customer.metadata.get('user_id')
        
        if user_id and db:
            user_ref = db.collection('Users').document(user_id)
            user_ref.update({
                'subscription.status': 'past_due',
                'subscription.updated_at': datetime.now().isoformat(),
            })
            
            print(f"⚠️ Invoice payment failed for user {user_id}")
    except Exception as e:
        print(f"❌ Error handling invoice payment failed: {e}")

