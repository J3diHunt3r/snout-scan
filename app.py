import os
import json

# Optional imports for face recognition (may not be available on all platforms)
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    print("⚠️ face_recognition not available - some features will be disabled")
    FACE_RECOGNITION_AVAILABLE = False
    face_recognition = None

from flask import Flask, jsonify, request
from PIL import Image, ExifTags
import torch
import numpy as np
import cv2
import uuid
import base64
import firebase_admin
from firebase_admin import credentials, firestore
from google.cloud import firestore as google_firestore
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity

# Optional dlib import (requires C++ compilation, may not be available)
try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    print("⚠️ dlib not available - using OpenCV for face detection instead")
    DLIB_AVAILABLE = False
    dlib = None

from scipy.spatial.distance import euclidean
import warnings
from dotenv import load_dotenv
import stripe
from datetime import datetime, timedelta
from flask_cors import CORS

# Machine Learning Enhancements - DogFaceNet Integration
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    from tensorflow.keras.applications import ResNet50
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Lambda
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
    print("✅ TensorFlow successfully imported for DogFaceNet")
except ImportError as e:
    print(f"⚠️ TensorFlow not available: {e}")
    print("   DogFaceNet features will be disabled")
    TENSORFLOW_AVAILABLE = False

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
try:
    firebase_admin.get_app()
    print("Firebase app already initialized")
except ValueError:
    try:
        if os.path.exists('firebase-credentials.json'):
            print("Using firebase-credentials.json")
            cred = credentials.Certificate('firebase-credentials.json')
            firebase_admin.initialize_app(cred)
        else:
            print("No credentials file found, trying default credentials")
            firebase_admin.initialize_app()
    except Exception as e:
        print(f"Firebase initialization error: {e}")
        db = None
    else:
        try:
            db = firestore.client()
            print("Firebase Firestore client initialized successfully")
        except Exception as e:
            print(f"Firestore client error: {e}")
            db = None

# Load models
try:
    # Try to use a more recent YOLOv5 loading approach
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)
    model.conf = 0.1  # Lower confidence threshold for better detection
    model.iou = 0.45  # Set IoU threshold for NMS
    print("YOLOv5 model loaded successfully")
except Exception as e:
    print(f"Error loading YOLOv5 model: {e}")
    # Fallback to basic model
    model = None

# Create directories
os.makedirs('uploads', exist_ok=True)
os.makedirs('snout_data', exist_ok=True)

# Load environment variables
load_dotenv()

# Initialize Stripe
stripe.api_key = os.getenv('STRIPE_SECRET_KEY', '')
if stripe.api_key:
    print("✅ Stripe initialized successfully")
else:
    print("⚠️ Stripe secret key not found. Payment features will be disabled.")

# Initialize Stripe checkout module with Firestore client
# Note: We import it as stripe_checkout_module to avoid conflict with stripe package
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
    import traceback
    traceback.print_exc()
    stripe_module = None


class ImprovedMuzzleFeatureExtractor:
    def __init__(self):
        if DLIB_AVAILABLE:
            self.detector = dlib.get_frontal_face_detector()
        else:
            self.detector = None  # Will use OpenCV fallback

    def extract_muzzle_region(self, image, animal_box):
        """Extract a more focused muzzle/nose region"""
        try:
            left, top, right, bottom = animal_box
            
            # Validate bounding box
            if left >= right or top >= bottom:
                print(f"Invalid bounding box: left={left}, top={top}, right={right}, bottom={bottom}")
                return None
            
            # Ensure coordinates are within image bounds
            img_height, img_width = image.shape[:2]
            left = max(0, min(left, img_width - 1))
            top = max(0, min(top, img_height - 1))
            right = max(left + 1, min(right, img_width))
            bottom = max(top + 1, min(bottom, img_height))
            
            # Check minimum size
            if (right - left) < 20 or (bottom - top) < 20:
                print(f"Region too small: width={right-left}, height={bottom-top}")
                return None
            
            animal_region = image[top:bottom, left:right]
            
            if animal_region.size == 0:
                print("Animal region is empty")
                return None
            
            # Convert to grayscale
            if len(animal_region.shape) == 3:
                gray_animal = cv2.cvtColor(animal_region, cv2.COLOR_RGB2GRAY)
            else:
                gray_animal = animal_region
                
            height, width = gray_animal.shape

            # More focused muzzle extraction - concentrate on nose/snout area
            # For dogs, focus on lower-center portion but much smaller region
            muzzle_top = int(height * 0.45)     # Start lower (45% from top)
            muzzle_bottom = int(height * 0.85)  # End higher (40% height instead of 60%)
            muzzle_left = int(width * 0.25)     # More centered (50% width instead of 80%)
            muzzle_right = int(width * 0.75)

            # Ensure valid coordinates
            muzzle_top = max(0, min(muzzle_top, height - 1))
            muzzle_bottom = max(muzzle_top + 1, min(muzzle_bottom, height))
            muzzle_left = max(0, min(muzzle_left, width - 1))
            muzzle_right = max(muzzle_left + 1, min(muzzle_right, width))

            muzzle_region = gray_animal[muzzle_top:muzzle_bottom, muzzle_left:muzzle_right]

            if muzzle_region.size == 0:
                print("Muzzle region is empty")
                return None

            print(f"Focused muzzle region extracted: {muzzle_region.shape}")
            return muzzle_region
            
        except Exception as e:
            print(f"Error in extract_muzzle_region: {e}")
            return None

    def extract_enhanced_texture_features(self, muzzle_region):
        """Extract more distinctive texture features"""
        if muzzle_region is None or muzzle_region.size == 0:
            return None

        try:
            # Resize to standard size with better interpolation
            muzzle_resized = cv2.resize(muzzle_region, (128, 128), interpolation=cv2.INTER_CUBIC)

            # Apply histogram equalization for better contrast
            muzzle_equalized = cv2.equalizeHist(muzzle_resized)

            # Calculate LBP with different radius and points for more detail
            lbp = self.calculate_enhanced_lbp(muzzle_equalized, radius=2, n_points=16)

            # Create histogram with more bins for better discrimination
            hist, _ = np.histogram(lbp.ravel(), bins=512, range=[0, 512])
            
            # Normalize but preserve relative magnitudes
            hist = hist.astype(float)
            hist = hist / (hist.sum() + 1e-8)  # Prevent division by zero
            
            # Add statistical features
            mean_intensity = np.mean(muzzle_equalized)
            std_intensity = np.std(muzzle_equalized)
            skewness = self.calculate_skewness(muzzle_equalized)
            
            # Combine histogram with statistical features
            enhanced_features = np.concatenate([
                hist, 
                [mean_intensity/255.0, std_intensity/255.0, skewness]
            ])
            
            return enhanced_features
            
        except Exception as e:
            print(f"Error in enhanced texture extraction: {e}")
            return None

    def calculate_enhanced_lbp(self, image, radius=2, n_points=16):
        """Calculate enhanced Local Binary Pattern with configurable parameters"""
        try:
            rows, cols = image.shape
            lbp = np.zeros((rows - 2*radius, cols - 2*radius))

            for i in range(radius, rows - radius):
                for j in range(radius, cols - radius):
                    center = image[i, j]
                    binary_val = 0

                    for point in range(n_points):
                        # Calculate neighbor coordinates
                        angle = 2 * np.pi * point / n_points
                        x = i + radius * np.cos(angle)
                        y = j + radius * np.sin(angle)
                        
                        # Bilinear interpolation for non-integer coordinates
                        x1, y1 = int(x), int(y)
                        x2, y2 = min(x1 + 1, rows - 1), min(y1 + 1, cols - 1)
                        
                        # Weights for interpolation
                        wx, wy = x - x1, y - y1
                        
                        # Interpolated pixel value
                        neighbor_val = (1 - wx) * (1 - wy) * image[x1, y1] + \
                                     wx * (1 - wy) * image[x2, y1] + \
                                     (1 - wx) * wy * image[x1, y2] + \
                                     wx * wy * image[x2, y2]

                        if neighbor_val >= center:
                            binary_val += 2 ** point

                    lbp[i - radius, j - radius] = binary_val

            return lbp
        except Exception as e:
            print(f"Error in enhanced LBP calculation: {e}")
            return self.calculate_lbp(image)  # Fallback to simple LBP

    def calculate_lbp(self, image):
        """Calculate Local Binary Pattern (fallback method)"""
        rows, cols = image.shape
        lbp = np.zeros((rows - 2, cols - 2))

        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                center = image[i, j]
                binary_string = ''

                # Check 8 neighbors
                neighbors = [
                    image[i - 1, j - 1], image[i - 1, j], image[i - 1, j + 1],
                    image[i, j + 1], image[i + 1, j + 1], image[i + 1, j],
                    image[i + 1, j - 1], image[i, j - 1]
                ]

                for neighbor in neighbors:
                    binary_string += '1' if neighbor >= center else '0'

                lbp[i - 1, j - 1] = int(binary_string, 2)

        return lbp

    def calculate_skewness(self, image):
        """Calculate skewness of image intensities"""
        try:
            flat_image = image.flatten()
            mean_val = np.mean(flat_image)
            std_val = np.std(flat_image)
            if std_val == 0:
                return 0
            skewness = np.mean(((flat_image - mean_val) / std_val) ** 3)
            return np.clip(skewness, -3, 3)  # Clip extreme values
        except:
            return 0

    def extract_multi_scale_features(self, muzzle_region):
        """Extract features at multiple scales for better discrimination"""
        if muzzle_region is None or muzzle_region.size == 0:
            return None

        try:
            features_list = []
            
            # Extract features at different scales
            scales = [(64, 64), (128, 128), (96, 96)]
            
            for scale in scales:
                resized = cv2.resize(muzzle_region, scale, interpolation=cv2.INTER_CUBIC)
                
                # Texture features
                texture_features = self.extract_enhanced_texture_features(resized)
                if texture_features is not None:
                    # Take only most significant features to avoid dimensionality explosion
                    features_list.append(texture_features[:100])  # Limit features per scale
            
            if not features_list:
                return None
                
            # Combine all scale features
            combined_features = np.concatenate(features_list)
            return combined_features
            
        except Exception as e:
            print(f"Error in multi-scale feature extraction: {e}")
            return None

    def extract_comprehensive_features(self, image, animal_box):
        """Extract comprehensive muzzle features with improved methods"""
        try:
            print("Starting comprehensive feature extraction...")
            
            # Get muzzle region
            muzzle_region = self.extract_muzzle_region(image, animal_box)
            if muzzle_region is None:
                print("Failed to extract muzzle region")
                return None

            print(f"Muzzle region extracted successfully, shape: {muzzle_region.shape}")

            # Extract enhanced texture features
            texture_features = self.extract_enhanced_texture_features(muzzle_region)
            if texture_features is None:
                print("Failed to extract enhanced texture features")
                return None
                
            print(f"Enhanced texture features extracted: {len(texture_features)}")

            # Extract multi-scale features
            multi_scale_features = self.extract_multi_scale_features(muzzle_region)
            if multi_scale_features is not None:
                print(f"Multi-scale features extracted: {len(multi_scale_features)}")
                # Combine with texture features
                combined_features = np.concatenate([texture_features, multi_scale_features])
            else:
                print("Multi-scale features failed, using texture features only")
                combined_features = texture_features

            print(f"Combined features length: {len(combined_features)}")

            # Store features without normalization to preserve uniqueness
            # Normalization was destroying the distinctive characteristics of each animal
            print("Features stored without normalization to preserve uniqueness")
            
            # Debug: Show actual feature values to verify uniqueness
            print(f"Feature sample (first 10): {combined_features[:10]}")
            print(f"Feature sample (last 10): {combined_features[-10:]}")
            print(f"Feature statistics - Min: {combined_features.min():.6f}, Max: {combined_features.max():.6f}, Mean: {combined_features.mean():.6f}")

            print("Feature extraction completed successfully")
            
            return {
                'features': combined_features.tolist(),
                'muzzle_size': muzzle_region.shape,
                'feature_dimension': len(combined_features),
                'texture_dim': len(texture_features),
                'multi_scale_dim': len(multi_scale_features) if multi_scale_features is not None else 0
            }
            
        except Exception as e:
            print(f"Error in extract_comprehensive_features: {e}")
            import traceback
            traceback.print_exc()
            return None


# Initialize feature extractor
feature_extractor = ImprovedMuzzleFeatureExtractor()


# Machine Learning Enhancement: DogFaceNet Integration
class DogFaceNetEnhancer:
    """
    Integration of DogFaceNet for improved dog face recognition
    Based on the FaceNet architecture specifically trained for dogs
    """
    
    def __init__(self):
        self.model = None
        self.model_path = "models/dogfacenet_model"
        self.embedding_size = 128
        self.input_size = (224, 224, 3)
        self.is_loaded = False
        
        if TENSORFLOW_AVAILABLE:
            self.initialize_model()
        else:
            print("⚠️ DogFaceNet disabled - TensorFlow not available")
    
    def initialize_model(self):
        """Initialize the DogFaceNet model"""
        try:
            print("🚀 Initializing DogFaceNet model...")
            
            # Create models directory
            os.makedirs('models', exist_ok=True)
            
            # Create DogFaceNet-inspired architecture
            self.create_dogfacenet_architecture()
            
            if self.is_loaded:
                print("✅ DogFaceNet model initialized successfully!")
            else:
                print("❌ DogFaceNet model initialization failed")
                
        except Exception as e:
            print(f"❌ Error initializing DogFaceNet: {e}")
            self.is_loaded = False
    
    def create_dogfacenet_architecture(self):
        """
        Create DogFaceNet-inspired architecture using ResNet backbone
        Based on the paper's methodology but adapted for TensorFlow 2.x
        """
        try:
            print("🔧 Creating DogFaceNet-inspired architecture...")
            
            # Use ResNet50 as backbone (similar to original paper)
            base_model = ResNet50(
                weights='imagenet', 
                include_top=False, 
                input_shape=self.input_size
            )
            
            # Freeze base model layers for transfer learning
            base_model.trainable = False
            
            # Add custom head for dog face embeddings
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(512, activation='relu', name='fc1')(x)
            x = Dense(256, activation='relu', name='fc2')(x)
            
            # Embedding layer (similar to FaceNet)
            embeddings = Dense(self.embedding_size, activation=None, name='embeddings')(x)
            
            # L2 normalize embeddings (crucial for FaceNet-style architectures)
            embeddings = Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(embeddings)
            
            self.model = Model(inputs=base_model.input, outputs=embeddings)
            
            # Compile model
            self.model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='mse'  # Mean squared error for embedding learning
            )
            
            print(f"✅ Model created with input shape: {self.input_size}")
            print(f"✅ Embedding size: {self.embedding_size}")
            print(f"✅ Total parameters: {self.model.count_params():,}")
            
            self.is_loaded = True
            return True
            
        except Exception as e:
            print(f"❌ Error creating DogFaceNet architecture: {e}")
            self.is_loaded = False
            return False
    
    def preprocess_dog_face(self, image, bbox):
        """
        Preprocess dog face region for DogFaceNet
        Applies similar preprocessing as mentioned in the paper
        """
        try:
            left, top, right, bottom = bbox
            
            # Extract face region
            face_region = image[top:bottom, left:right]
            
            if face_region.size == 0:
                return None
            
            # Resize to model input size
            face_resized = cv2.resize(face_region, (self.input_size[1], self.input_size[0]))
            
            # Apply histogram equalization for better contrast
            if len(face_resized.shape) == 3:
                # Convert to LAB color space for better equalization
                lab = cv2.cvtColor(face_resized, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                l = clahe.apply(l)
                face_resized = cv2.merge([l, a, b])
                face_resized = cv2.cvtColor(face_resized, cv2.COLOR_LAB2RGB)
            
            # Normalize pixel values to [-1, 1] (common for face recognition models)
            face_normalized = (face_resized.astype(np.float32) - 127.5) / 127.5
            
            # Add batch dimension
            face_batch = np.expand_dims(face_normalized, axis=0)
            
            return face_batch
            
        except Exception as e:
            print(f"❌ Error preprocessing dog face: {e}")
            return None
    
    def extract_dog_embeddings(self, image, bbox):
        """
        Extract DogFaceNet embeddings from dog face region
        """
        try:
            if not self.is_loaded:
                print("⚠️ DogFaceNet model not loaded")
                return None
            
            # Preprocess face
            face_batch = self.preprocess_dog_face(image, bbox)
            if face_batch is None:
                return None
            
            # Extract embeddings
            embeddings = self.model.predict(face_batch, verbose=0)
            
            # Return flattened embeddings
            return embeddings[0].tolist()
            
        except Exception as e:
            print(f"❌ Error extracting dog embeddings: {e}")
            return None
    
    def calculate_dogfacenet_similarity(self, embeddings1, embeddings2):
        """
        Calculate similarity between DogFaceNet embeddings using cosine similarity
        """
        try:
            if not embeddings1 or not embeddings2:
                return 0.0, False
            
            # Convert to numpy arrays
            emb1 = np.array(embeddings1)
            emb2 = np.array(embeddings2)
            
            # Calculate cosine similarity (embeddings are already L2 normalized)
            similarity = np.dot(emb1, emb2)
            
            # Convert to positive range [0, 1]
            similarity = (similarity + 1) / 2
            
            # Threshold for match (can be adjusted)
            threshold = 0.85
            is_match = similarity >= threshold
            
            print(f"🐕 DogFaceNet similarity: {similarity:.4f}, match: {is_match}")
            
            return float(similarity), is_match
            
        except Exception as e:
            print(f"❌ Error calculating DogFaceNet similarity: {e}")
            return 0.0, False


def test_yolo_model():
    """Test function to verify YOLOv5 model is working correctly"""
    try:
        if model is None:
            print("YOLOv5 model is None - cannot test")
            return False
        
        # Create a simple test image (black square)
        test_image = np.zeros((640, 640, 3), dtype=np.uint8)
        
        print("Testing YOLOv5 model with simple test image...")
        results = model(test_image)
        
        print(f"Test results shape: {len(results.xyxy)}")
        print(f"Test predictions shape: {results.xyxy[0].shape if len(results.xyxy) > 0 else 'No predictions'}")
        print(f"Available classes: {list(results.names.values())}")
        
        return True
        
    except Exception as e:
        print(f"Error testing YOLOv5 model: {e}")
        import traceback
        traceback.print_exc()
        return False


# Test the model on startup
print("Testing YOLOv5 model...")
test_yolo_model()


# Quick SIFT-based improvement for your existing system
# Add this to your app.py to significantly improve accuracy

class QuickSIFTEnhancement:
    def __init__(self):
        """Initialize SIFT detector for keypoint matching"""
        try:
            # Try to create SIFT detector
            self.sift = cv2.SIFT_create(nfeatures=100)  # Reduced from 300 to stay within Firestore limits
            print("SIFT detector initialized successfully")
        except:
            # Fallback to ORB if SIFT is not available
            self.sift = cv2.ORB_create(nfeatures=100)  # Reduced from 300 to stay within Firestore limits
            print("Using ORB detector as SIFT fallback")
        
        # Initialize FLANN matcher for fast matching
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def extract_sift_features_from_muzzle(self, image, animal_box):
        """Extract SIFT keypoints from the muzzle region - FIRESTORE SAFE VERSION"""
        return extract_sift_features_firestore_safe(self, image, animal_box)

    def calculate_sift_similarity(self, features1, features2, threshold=0.75):
        """Calculate similarity based on SIFT keypoint matching"""
        try:
            if not features1 or not features2:
                return 0.0, False
            
            desc1 = features1['descriptors']
            desc2 = features2['descriptors']
            
            if desc1 is None or desc2 is None or len(desc1) < 5 or len(desc2) < 5:
                return 0.0, False
            
            # Convert lists back to numpy arrays for OpenCV matching
            desc1_array = np.array(desc1, dtype=np.float32)
            desc2_array = np.array(desc2, dtype=np.float32)
            
            # Match descriptors using FLANN
            try:
                matches = self.flann.knnMatch(desc1_array, desc2_array, k=2)
            except:
                # Fallback to brute force matching
                bf = cv2.BFMatcher()
                matches = bf.knnMatch(desc1_array, desc2_array, k=2)
            
            # Apply Lowe's ratio test to filter good matches
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:  # Lowe's ratio
                        good_matches.append(m)
            
            # Calculate similarity based on number of good matches
            total_possible_matches = min(len(desc1), len(desc2))
            match_ratio = len(good_matches) / total_possible_matches
            
            # Additional quality check - average distance of good matches
            if good_matches:
                avg_distance = np.mean([m.distance for m in good_matches])
                distance_quality = max(0, 1 - (avg_distance / 256))  # Normalize distance
                
                # Combine match ratio and distance quality
                similarity_score = 0.7 * match_ratio + 0.3 * distance_quality
            else:
                similarity_score = 0.0
            
            is_match = similarity_score >= threshold
            
            print(f"SIFT matching: {len(good_matches)} good matches out of {total_possible_matches} possible")
            print(f"Match ratio: {match_ratio:.3f}, Similarity: {similarity_score:.3f}, Match: {is_match}")
            
            return similarity_score, is_match
            
        except Exception as e:
            print(f"Error in SIFT similarity calculation: {e}")
            return 0.0, False


def extract_sift_features_firestore_safe(sift_enhancer, image, animal_box):
    """Extract SIFT keypoints from the muzzle region - FIRESTORE SAFE VERSION"""
    try:
        left, top, right, bottom = animal_box
        
        # Extract muzzle region
        animal_region = image[top:bottom, left:right]
        
        if len(animal_region.shape) == 3:
            gray_region = cv2.cvtColor(animal_region, cv2.COLOR_RGB2GRAY)
        else:
            gray_region = animal_region
        
        # Focus on the muzzle area
        height, width = gray_region.shape
        muzzle_top = int(height * 0.4)
        muzzle_bottom = int(height * 0.9)
        muzzle_left = int(width * 0.2)
        muzzle_right = int(width * 0.8)
        
        muzzle_region = gray_region[muzzle_top:muzzle_bottom, muzzle_left:muzzle_right]
        
        if muzzle_region.size == 0:
            return None
        
        # Resize to standard size for consistent matching
        muzzle_resized = cv2.resize(muzzle_region, (200, 200))
        
        # Apply histogram equalization for better feature detection
        muzzle_enhanced = cv2.equalizeHist(muzzle_resized)
        
        # Detect keypoints and compute descriptors
        keypoints, descriptors = sift_enhancer.sift.detectAndCompute(muzzle_enhanced, None)
        
        if descriptors is not None and len(descriptors) > 5:  # Need at least 5 keypoints
            print(f"Extracted {len(keypoints)} SIFT keypoints")
            
            # LIMIT SIFT FEATURES TO STAY WITHIN FIRESTORE LIMITS
            max_features = 100  # Reduced from 300 to stay under 1MB
            if len(descriptors) > max_features:
                # Take evenly distributed features
                step = len(descriptors) // max_features
                descriptors = descriptors[::step][:max_features]
                print(f"Limited SIFT features to {len(descriptors)} to stay within Firestore limits")
            
            # Convert numpy array to list for Firestore compatibility
            descriptors_list = descriptors.astype(np.float32).tolist()
            
            return {
                'keypoints_count': len(keypoints),
                'descriptors': descriptors_list,  # Now a Python list instead of numpy array
                'descriptor_dimension': descriptors.shape[1] if len(descriptors) > 0 else 0,
                'feature_count': len(descriptors)
            }
        else:
            print("Not enough keypoints detected")
            return None
            
    except Exception as e:
        print(f"Error extracting SIFT features: {e}")
        return None


# Integration with your existing system
sift_enhancer = QuickSIFTEnhancement()

# STEP 1: Modify your feature extraction in detect_animals_and_extract_features
def extract_enhanced_muzzle_features(image, animal_box):
    """Extract both your existing features AND SIFT features"""
    try:
        # Keep your existing feature extraction
        existing_features = feature_extractor.extract_comprehensive_features(image, animal_box)
        
        # Add SIFT features
        sift_features = sift_enhancer.extract_sift_features_from_muzzle(image, animal_box)
        
        if existing_features is not None and sift_features is not None:
            # Combine both feature types
            combined_features = {
                'traditional_features': existing_features,
                'sift_features': sift_features,
                'has_both': True
            }
            return combined_features
        elif existing_features is not None:
            # Fall back to existing features only
            return {
                'traditional_features': existing_features,
                'sift_features': None,
                'has_both': False
            }
        else:
            return None
            
    except Exception as e:
        print(f"Error in enhanced feature extraction: {e}")
        return None

# STEP 2: Modify your similarity calculation
def calculate_enhanced_similarity(features1, features2, threshold=0.75):
    """Calculate similarity using both traditional and SIFT features"""
    try:
        print(f"\n🔍 ENHANCED SIMILARITY CALCULATION:")
        print(f"   " + "-" * 40)
        
        if not features1 or not features2:
            print("   ❌ One or both feature sets are empty")
            return 0.0, False
        
        print(f"   ✅ Both feature sets present")
        print(f"   Features1 type: {type(features1)}")
        print(f"   Features2 type: {type(features2)}")
        
        # Normalize both feature sets to ensure consistent structure
        print(f"   🔧 Normalizing features...")
        normalized_features1 = normalize_features_for_comparison(features1)
        normalized_features2 = normalize_features_for_comparison(features2)
        
        if not normalized_features1 or not normalized_features2:
            print("   ❌ Failed to normalize features for comparison")
            return 0.0, False
        
        print(f"   ✅ Features normalized successfully")
        print(f"   Normalized Features1 keys: {list(normalized_features1.keys())}")
        print(f"   Normalized Features2 keys: {list(normalized_features2.keys())}")
        
        # Calculate traditional similarity (your existing method)
        traditional_sim = 0.0
        traditional_match = False
        
        if (normalized_features1.get('traditional_features') and 
            normalized_features2.get('traditional_features')):
            print("Both have traditional features, calculating similarity...")
            traditional_sim, traditional_match = calculate_improved_similarity(
                normalized_features1['traditional_features'], 
                normalized_features2['traditional_features'],
                threshold
            )
            print(f"Traditional similarity result: {traditional_sim:.3f}, match: {traditional_match}")
        else:
            print("Missing traditional features in one or both sets")
        
        # Calculate SIFT similarity
        sift_sim = 0.0
        sift_match = False
        
        if (normalized_features1.get('sift_features') and 
            normalized_features2.get('sift_features')):
            print("Both have SIFT features, calculating similarity...")
            sift_sim, sift_match = calculate_sift_similarity_safe(
                normalized_features1,
                normalized_features2,
                threshold
            )
            print(f"SIFT similarity result: {sift_sim:.3f}, match: {sift_match}")
        else:
            print("Missing SIFT features in one or both sets")
        
        # Combine similarities with weights
        if normalized_features1.get('has_both') and normalized_features2.get('has_both'):
            # Both have SIFT and traditional features - use both
            combined_similarity = 0.6 * sift_sim + 0.4 * traditional_sim
            combined_match = combined_similarity >= threshold
            print(f"Combined similarity: SIFT={sift_sim:.3f}, Traditional={traditional_sim:.3f}, Combined={combined_similarity:.3f}")
        elif sift_sim > 0:
            # Only SIFT available
            combined_similarity = sift_sim
            combined_match = sift_match
            print(f"SIFT-only similarity: {sift_sim:.3f}")
        else:
            # Only traditional features
            combined_similarity = traditional_sim
            combined_match = traditional_match
            print(f"Traditional-only similarity: {traditional_sim:.3f}")
        
        return combined_similarity, combined_match
        
    except Exception as e:
        print(f"Error in enhanced similarity calculation: {e}")
        import traceback
        traceback.print_exc()
        return 0.0, False


# Add this debugging function
def validate_and_debug_features(animal_data, pet_name="Unknown"):
    """Debug function to validate extracted features"""
    print(f"\n=== Feature Validation for {pet_name} ===")
    for i, animal in enumerate(animal_data):
        print(f"Animal {i+1}: {animal['animal_type']}")
        if 'muzzle_features' in animal:
            muzzle_features = animal['muzzle_features']
            
            # Check if this is the new enhanced structure
            if 'traditional_features' in muzzle_features:
                print("  Enhanced feature structure detected:")
                print(f"  Has both features: {muzzle_features.get('has_both', False)}")
                
                # Validate traditional features
                if 'traditional_features' in muzzle_features and muzzle_features['traditional_features']:
                    trad_features = muzzle_features['traditional_features']['features']
                    print(f"  Traditional feature vector length: {len(trad_features)}")
                    print(f"  Traditional feature range: [{min(trad_features):.4f}, {max(trad_features):.4f}]")
                    print(f"  Traditional feature mean: {np.mean(trad_features):.4f}")
                    print(f"  Traditional feature std: {np.std(trad_features):.4f}")
                
                # Validate SIFT features
                if 'sift_features' in muzzle_features and muzzle_features['sift_features']:
                    sift_features = muzzle_features['sift_features']
                    print(f"  SIFT keypoints count: {sift_features['keypoints_count']}")
                    
                    # Handle both numpy array and list formats
                    descriptors = sift_features['descriptors']
                    if hasattr(descriptors, 'shape'):
                        # It's a numpy array
                        print(f"  SIFT descriptors shape: {descriptors.shape}")
                    else:
                        # It's a list
                        print(f"  SIFT descriptors shape: ({len(descriptors)}, {len(descriptors[0]) if descriptors else 0})")
                else:
                    print("  No SIFT features found")
                    
            else:
                # Old structure (fallback compatibility)
                features = muzzle_features['features']
                print(f"  Feature vector length: {len(features)}")
                print(f"  Feature range: [{min(features):.4f}, {max(features):.4f}]")
                print(f"  Feature mean: {np.mean(features):.4f}")
                print(f"  Feature std: {np.std(features):.4f}")
                print(f"  Zero values: {np.sum(np.array(features) == 0)}")
                print(f"  NaN values: {np.sum(np.isnan(features))}")
        else:
            print("  No muzzle features found!")
    print("=" * 50)


def normalize_features_for_comparison(features):
    """Normalize features to ensure consistent structure for comparison"""
    try:
        if not features:
            return None
            
        # If it's already the new enhanced structure, return as is
        if 'traditional_features' in features:
            return features
            
        # If it's the old structure, convert to new structure
        if 'features' in features:
            return {
                'traditional_features': features,
                'sift_features': None,
                'has_both': False
            }
            
        # Unknown structure
        print(f"Unknown feature structure: {list(features.keys())}")
        return None
        
    except Exception as e:
        print(f"Error normalizing features: {e}")
        return None


def calculate_improved_similarity(features1, features2, threshold=0.85):
    """Improved similarity calculation with multiple metrics"""
    try:
        if not features1 or not features2:
            return 0.0, False

        f1 = np.array(features1['features'])
        f2 = np.array(features2['features'])
        
        # Ensure features are valid
        if len(f1) != len(f2):
            print(f"Feature dimension mismatch: {len(f1)} vs {len(f2)}")
            return 0.0, False
        
        if np.any(np.isnan(f1)) or np.any(np.isnan(f2)):
            print("NaN values detected in features")
            return 0.0, False
        
        # Multiple similarity metrics
        
        # 1. Cosine similarity (normalized dot product)
        norm1 = np.linalg.norm(f1)
        norm2 = np.linalg.norm(f2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0, False
            
        # Normalize for cosine similarity calculation only
        f1_norm = f1 / norm1
        f2_norm = f2 / norm2
        cosine_sim = np.dot(f1_norm, f2_norm)
        
        # 2. Correlation coefficient
        if np.std(f1) > 0 and np.std(f2) > 0:
            correlation = np.corrcoef(f1, f2)[0, 1]
            correlation = 0 if np.isnan(correlation) else correlation
        else:
            correlation = 0
        
        # 3. Normalized euclidean distance
        euclidean_dist = np.linalg.norm(f1 - f2)
        max_possible_dist = np.sqrt(2 * len(f1))  # Maximum possible distance for normalized vectors
        euclidean_sim = 1 - (euclidean_dist / max_possible_dist)
        
        # 4. Chi-square distance for histogram-like features
        chi_square_sim = 0
        if np.all(f1 >= 0) and np.all(f2 >= 0):  # Only for non-negative features
            chi_square_dist = np.sum((f1 - f2)**2 / (f1 + f2 + 1e-8))
            chi_square_sim = 1 / (1 + chi_square_dist)
        
        # Weighted combination of similarities
        combined_similarity = (
            0.4 * cosine_sim +           # Primary metric
            0.25 * euclidean_sim +       # Distance-based
            0.2 * abs(correlation) +     # Correlation
            0.15 * chi_square_sim        # Distribution similarity
        )
        
        # Apply stricter threshold
        is_match = combined_similarity >= threshold
        
        print(f"Similarity breakdown - Cosine: {cosine_sim:.3f}, "
              f"Euclidean: {euclidean_sim:.3f}, Correlation: {correlation:.3f}, "
              f"Chi-square: {chi_square_sim:.3f}, Combined: {combined_similarity:.3f}")
        
        return combined_similarity, is_match

    except Exception as e:
        print(f"Error calculating improved similarity: {e}")
        return 0.0, False


def preprocess_image_for_better_detection(image):
    """Enhanced image preprocessing for better animal detection"""
    try:
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l_clahe = clahe.apply(l)
            
            # Merge channels and convert back
            lab_clahe = cv2.merge([l_clahe, a, b])
            enhanced_image = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced_image = clahe.apply(image)
        
        # Apply slight gaussian blur to reduce noise
        enhanced_image = cv2.GaussianBlur(enhanced_image, (3, 3), 0)
        
        return enhanced_image
        
    except Exception as e:
        print(f"Error in image preprocessing: {e}")
        return image


def clean_data_for_firestore(data, path="root"):
    """Clean data to make it compatible with Firestore by converting numpy arrays to lists"""
    try:
        if isinstance(data, dict):
            cleaned = {}
            for key, value in data.items():
                current_path = f"{path}.{key}"
                cleaned_value = clean_data_for_firestore(value, current_path)
                if cleaned_value is not None:  # Skip None values
                    cleaned[key] = cleaned_value
            return cleaned
        elif isinstance(data, (list, tuple)):
            cleaned_list = []
            for i, item in enumerate(data):
                current_path = f"{path}[{i}]"
                cleaned_item = clean_data_for_firestore(item, current_path)
                if cleaned_item is not None:  # Skip None values
                    cleaned_list.append(cleaned_item)
            return cleaned_list
        elif isinstance(data, np.ndarray):
            print(f"Converting numpy array at {path}: shape {data.shape}, dtype {data.dtype}")
            # Check array size to avoid huge data
            if data.size > 100000:  # Skip very large arrays
                print(f"Skipping large array at {path} with size {data.size}")
                return None
            return data.tolist()
        elif isinstance(data, np.integer):
            return int(data)
        elif isinstance(data, np.floating):
            return float(data)
        elif isinstance(data, (str, int, float, bool)):
            return data
        elif data is None:
            return None
        else:
            # Skip unsupported types
            print(f"Skipping unsupported type {type(data)} at {path}")
            return None
    except Exception as e:
        print(f"Error cleaning data at {path}: {e}")
        return None


def store_pet_safely(db, snout_id, pet_name, pet_breed, owner_name, animal_data, file_path):
    """Safely store pet data in Firestore with comprehensive data cleaning"""
    try:
        print("Starting safe pet storage...")
        
        # Clean the data to make it Firestore compatible
        print("Cleaning animal data for Firestore...")
        cleaned_animal_data = clean_data_for_firestore_comprehensive(animal_data)
        
        if cleaned_animal_data is None:
            print("Data cleaning failed - cannot store")
            return False
        
        print("Data cleaning completed successfully")
        
        # Create pet document
        pet_doc = {
            'snout_id': snout_id,
            'name': pet_name,
            'breed': pet_breed,
            'owner': owner_name,
            'muzzle_data': cleaned_animal_data,
            'created_at': firestore.SERVER_TIMESTAMP,
            'image_path': file_path,
            'storage_timestamp': firestore.SERVER_TIMESTAMP
        }
        
        # Validate document size
        doc_size = len(str(pet_doc))
        print(f"Document size: {doc_size} characters")
        
        if doc_size > 1000000:  # 1MB limit
            print("WARNING: Document is very large, may cause storage issues")
        
        # Store in Firestore
        print("Storing document in Firestore...")
        db.collection('my_pets').document(snout_id).set(pet_doc)
        
        print(f"Pet data successfully stored in Firebase with ID: {snout_id}")
        return True
        
    except Exception as e:
        print(f"Error in safe pet storage: {e}")
        import traceback
        traceback.print_exc()
        return False


def clean_data_for_firestore_robust(data, path="root", max_depth=20):
    """
    Robust data cleaning for Firestore compatibility with comprehensive type handling
    """
    if max_depth <= 0:
        print(f"Max depth reached at {path}, skipping")
        return None
        
    try:
        # Handle None values
        if data is None:
            return None
            
        # Handle dictionaries
        elif isinstance(data, dict):
            cleaned = {}
            for key, value in data.items():
                # Ensure key is string
                if not isinstance(key, str):
                    key = str(key)
                
                cleaned_value = clean_data_for_firestore_robust(value, f"{path}.{key}", max_depth - 1)
                if cleaned_value is not None:  # Only add non-None values
                    cleaned[key] = cleaned_value
            return cleaned
            
        # Handle lists and tuples
        elif isinstance(data, (list, tuple)):
            cleaned_list = []
            for i, item in enumerate(data):
                cleaned_item = clean_data_for_firestore_robust(item, f"{path}[{i}]", max_depth - 1)
                if cleaned_item is not None:
                    cleaned_list.append(cleaned_item)
            return cleaned_list
            
        # Handle numpy arrays
        elif isinstance(data, np.ndarray):
            print(f"Converting numpy array at {path}: shape {data.shape}, dtype {data.dtype}")
            
            # Check for reasonable array sizes
            if data.size > 100000:  # Skip very large arrays
                print(f"Skipping large array at {path} with size {data.size}")
                return f"[Large array of size {data.size} - skipped for storage]"
            
            # Convert to list and clean recursively
            array_as_list = data.tolist()
            return clean_data_for_firestore_robust(array_as_list, f"{path}_array", max_depth - 1)
            
        # Handle numpy scalar types
        elif isinstance(data, (np.integer, np.floating)):
            return data.item()  # Convert to Python native type
            
        # Handle numpy bool
        elif isinstance(data, np.bool_):
            return bool(data)
            
        # Handle Python native types (already compatible)
        elif isinstance(data, (str, int, float, bool)):
            return data
            
        # Handle bytes
        elif isinstance(data, bytes):
            try:
                return data.decode('utf-8')
            except:
                return f"[Binary data of length {len(data)}]"
                
        # Handle any other object by converting to string representation
        else:
            print(f"Converting unsupported type {type(data)} at {path} to string")
            return str(data)
            
    except Exception as e:
        print(f"Error cleaning data at {path}: {e}")
        return f"[Error processing data: {str(e)}]"


def validate_firestore_document(doc_data, max_size_mb=1.0):
    """
    Validate that a document can be stored in Firestore
    """
    try:
        # Convert to JSON string to estimate size
        import json
        doc_json = json.dumps(doc_data, default=str)
        doc_size_bytes = len(doc_json.encode('utf-8'))
        doc_size_mb = doc_size_bytes / (1024 * 1024)
        
        print(f"Document size: {doc_size_mb:.2f} MB ({doc_size_bytes} bytes)")
        
        if doc_size_mb > max_size_mb:
            print(f"WARNING: Document exceeds {max_size_mb} MB limit")
            return False, f"Document too large: {doc_size_mb:.2f} MB"
        
        # Check for nested depth
        def check_depth(obj, current_depth=0, max_allowed_depth=20):
            if current_depth > max_allowed_depth:
                return False
            
            if isinstance(obj, dict):
                return all(check_depth(v, current_depth + 1, max_allowed_depth) for v in obj.values())
            elif isinstance(obj, list):
                return all(check_depth(item, current_depth + 1, max_allowed_depth) for item in obj)
            else:
                return True
        
        if not check_depth(doc_data):
            return False, "Document nesting too deep"
        
        return True, "Document valid for Firestore"
        
    except Exception as e:
        return False, f"Validation error: {e}"


def reduce_data_size(animal_data):
    """
    Reduce data size by limiting feature dimensions and removing less critical data
    """
    try:
        if not isinstance(animal_data, list):
            return animal_data
        
        reduced_data = []
        
        for animal in animal_data:
            if not isinstance(animal, dict):
                reduced_data.append(animal)
                continue
                
            reduced_animal = animal.copy()
            
            # Reduce muzzle features if present
            if 'muzzle_features' in reduced_animal:
                mf = reduced_animal['muzzle_features']
                
                # Limit SIFT features
                if isinstance(mf, dict) and 'sift_features' in mf and mf['sift_features']:
                    sift_data = mf['sift_features']
                    if isinstance(sift_data, dict) and 'descriptors' in sift_data:
                        descriptors = sift_data['descriptors']
                        if isinstance(descriptors, list) and len(descriptors) > 50:
                            # Keep only first 50 descriptors
                            sift_data['descriptors'] = descriptors[:50]
                            sift_data['keypoints_count'] = min(sift_data.get('keypoints_count', 0), 50)
                            print(f"Reduced SIFT descriptors from {len(descriptors)} to 50")
                
                # Limit traditional features
                if isinstance(mf, dict) and 'traditional_features' in mf and mf['traditional_features']:
                    trad_data = mf['traditional_features']
                    if isinstance(trad_data, dict) and 'features' in trad_data:
                        features = trad_data['features']
                        if isinstance(features, list) and len(features) > 500:
                            # Keep only first 500 features
                            trad_data['features'] = features[:500]
                            trad_data['feature_dimension'] = min(trad_data.get('feature_dimension', 0), 500)
                            print(f"Reduced traditional features from {len(features)} to 500")
            
            reduced_data.append(reduced_animal)
        
        return reduced_data
        
    except Exception as e:
        print(f"Error reducing data size: {e}")
        return animal_data


import json
import copy

def deep_debug_data_structure(data, path="root", level=0):
    """
    Deep debugging to find the exact problematic structure
    """
    indent = "  " * level
    print(f"{indent}{path}: {type(data)}")
    
    try:
        if isinstance(data, dict):
            if level < 5:  # Prevent too deep recursion in logs
                for key, value in data.items():
                    deep_debug_data_structure(value, f"{path}.{key}", level + 1)
            else:
                print(f"{indent}  ... (dict with {len(data)} keys - too deep to show)")
                
        elif isinstance(data, (list, tuple)):
            print(f"{indent}  Length: {len(data)}")
            if level < 3 and len(data) > 0:  # Only show first few items
                for i, item in enumerate(data[:3]):
                    deep_debug_data_structure(item, f"{path}[{i}]", level + 1)
                if len(data) > 3:
                    print(f"{indent}  ... ({len(data) - 3} more items)")
                    
        elif isinstance(data, np.ndarray):
            print(f"{indent}  Numpy array: shape={data.shape}, dtype={data.dtype}")
            
        else:
            # Show actual value for simple types
            if isinstance(data, (str, int, float, bool)) and len(str(data)) < 50:
                print(f"{indent}  Value: {data}")
            else:
                print(f"{indent}  Type: {type(data)}, Length: {len(str(data))}")
                
    except Exception as e:
        print(f"{indent}  ERROR debugging this item: {e}")


def ultra_aggressive_firestore_cleaner(data, path="root", max_depth=15):
    """
    Ultra aggressive cleaner that handles ALL edge cases for Firestore
    """
    if max_depth <= 0:
        print(f"Max depth reached at {path}")
        return "[MAX_DEPTH_REACHED]"
        
    try:
        # Handle None
        if data is None:
            return None
            
        # Handle basic Python types (already Firestore compatible)
        elif isinstance(data, (str, int, float, bool)):
            # Validate values
            if isinstance(data, float):
                if np.isnan(data) or np.isinf(data):
                    return 0.0
            return data
            
        # Handle dictionaries
        elif isinstance(data, dict):
            cleaned = {}
            for key, value in data.items():
                # Ensure key is a valid string
                if not isinstance(key, str):
                    key = str(key)
                
                # Clean key (remove special characters that might cause issues)
                clean_key = ''.join(c for c in key if c.isalnum() or c in '_-.')[:100]  # Limit key length
                
                cleaned_value = ultra_aggressive_firestore_cleaner(
                    value, f"{path}.{clean_key}", max_depth - 1
                )
                
                # Only add if not None and not error markers
                if cleaned_value is not None and not isinstance(cleaned_value, str) or not cleaned_value.startswith('['):
                    cleaned[clean_key] = cleaned_value
                    
            return cleaned
            
        # Handle lists and tuples
        elif isinstance(data, (list, tuple)):
            cleaned_list = []
            for i, item in enumerate(data):
                if i >= 1000:  # Limit array size
                    print(f"Truncating large array at {path} (size: {len(data)})")
                    break
                    
                cleaned_item = ultra_aggressive_firestore_cleaner(
                    item, f"{path}[{i}]", max_depth - 1
                )
                
                # Only add valid items
                if cleaned_item is not None:
                    cleaned_list.append(cleaned_item)
                    
            return cleaned_list
            
        # Handle numpy arrays
        elif isinstance(data, np.ndarray):
            print(f"Processing numpy array at {path}: {data.shape}")
            
            # Handle different array types
            if data.size == 0:
                return []
                
            if data.size > 50000:  # Very large arrays
                print(f"Skipping very large array at {path}")
                return f"[LARGE_ARRAY_{data.shape}]"
                
            # Convert to list and clean recursively
            try:
                as_list = data.tolist()
                return ultra_aggressive_firestore_cleaner(as_list, f"{path}_array", max_depth - 1)
            except Exception as e:
                print(f"Error converting array to list at {path}: {e}")
                return f"[ARRAY_CONVERSION_ERROR]"
                
        # Handle numpy scalar types
        elif hasattr(data, 'item'):  # numpy scalars
            return data.item()
            
        # Handle other types by converting to string
        else:
            print(f"Converting unknown type {type(data)} to string at {path}")
            return str(data)[:1000]  # Limit string length
            
    except Exception as e:
        print(f"Error cleaning data at {path}: {e}")
        return f"[CLEAN_ERROR: {str(e)[:100]}]"


def test_firestore_compatibility(data):
    """
    Test if data can be serialized to JSON (Firestore requirement)
    """
    try:
        # Try to serialize to JSON
        json_str = json.dumps(data, default=str, ensure_ascii=False)
        print(f"JSON serialization successful, size: {len(json_str)} chars")
        
        # Try to deserialize back
        parsed_back = json.loads(json_str)
        print("JSON round-trip successful")
        
        return True, "Compatible with Firestore"
        
    except Exception as e:
        print(f"JSON serialization failed: {e}")
        return False, str(e)


def store_pet_with_ultra_cleaning(db, snout_id, pet_name, pet_breed, owner_name, animal_data, file_path):
    """
    Store pet with ultra-aggressive cleaning and debugging
    """
    try:
        print("\n" + "="*60)
        print("STARTING ULTRA CLEANING AND STORAGE")
        print("="*60)
        
        # Step 1: Debug the original data structure
        print("\n--- DEBUGGING ORIGINAL DATA STRUCTURE ---")
        deep_debug_data_structure(animal_data, "original_animal_data")
        
        # Step 2: Apply ultra cleaning
        print("\n--- APPLYING ULTRA AGGRESSIVE CLEANING ---")
        cleaned_animal_data = ultra_aggressive_firestore_cleaner(animal_data, "animal_data")
        
        print("\n--- DEBUGGING CLEANED DATA STRUCTURE ---")
        deep_debug_data_structure(cleaned_animal_data, "cleaned_animal_data")
        
        # Step 3: Test Firestore compatibility
        print("\n--- TESTING FIRESTORE COMPATIBILITY ---")
        is_compatible, compatibility_msg = test_firestore_compatibility(cleaned_animal_data)
        print(f"Compatibility test: {compatibility_msg}")
        
        if not is_compatible:
            print("Data still not compatible - applying emergency flattening")
            cleaned_animal_data = emergency_flatten_data(cleaned_animal_data)
            
            # Test again
            is_compatible, compatibility_msg = test_firestore_compatibility(cleaned_animal_data)
            print(f"Emergency flattening result: {compatibility_msg}")
        
        # Step 4: Create minimal document first
        minimal_doc = {
            'snout_id': snout_id,
            'name': pet_name,
            'breed': pet_breed,
            'owner': owner_name,
            'image_path': file_path,
            'created_at': firestore.SERVER_TIMESTAMP,
            'animals_detected': len(animal_data) if isinstance(animal_data, list) else 1,
            'processing_status': 'cleaned_with_ultra_method'
        }
        
        # Test minimal document
        print("\n--- TESTING MINIMAL DOCUMENT ---")
        is_compatible, compatibility_msg = test_firestore_compatibility(minimal_doc)
        print(f"Minimal document compatibility: {compatibility_msg}")
        
        if is_compatible:
            print("\n--- STORING MINIMAL DOCUMENT FIRST ---")
            db.collection('my_pets').document(snout_id).set(minimal_doc)
            print("Minimal document stored successfully!")
            
            # Now try to add the muzzle data
            print("\n--- ATTEMPTING TO UPDATE WITH MUZZLE DATA ---")
            try:
                db.collection('my_pets').document(snout_id).update({
                    'muzzle_data': cleaned_animal_data,
                    'muzzle_data_added': firestore.SERVER_TIMESTAMP
                })
                print("Muzzle data added successfully!")
                return True
                
            except Exception as muzzle_error:
                print(f"Muzzle data update failed: {muzzle_error}")
                print("But minimal pet info was saved!")
                
                # Store error info
                db.collection('my_pets').document(snout_id).update({
                    'muzzle_data_error': str(muzzle_error)[:500],
                    'error_timestamp': firestore.SERVER_TIMESTAMP
                })
                
                return False
        else:
            print("Even minimal document failed - this is a serious issue")
            return False
            
    except Exception as e:
        print(f"Ultra cleaning storage failed: {e}")
        import traceback
        traceback.print_exc()
        return False


import base64
import pickle

def flatten_nested_arrays(data, path="root"):
    """
    Specifically target and flatten nested arrays that cause Firestore issues
    """
    try:
        if isinstance(data, dict):
            flattened = {}
            for key, value in data.items():
                flattened[key] = flatten_nested_arrays(value, f"{path}.{key}")
            return flattened
            
        elif isinstance(data, list):
            # Check if this is a list of lists (nested array)
            if len(data) > 0 and isinstance(data[0], list):
                print(f"Found nested array at {path} with {len(data)} outer items")
                
                # Special handling for SIFT descriptors (list of 128-element lists)
                if all(isinstance(item, list) and len(item) == 128 for item in data[:5]):  # Check first few
                    print(f"Detected SIFT descriptors at {path}")
                    # Convert to flat array with metadata
                    flat_data = []
                    for descriptor in data:
                        flat_data.extend(descriptor)  # Flatten each 128-element array
                    
                    return {
                        'flattened_descriptors': flat_data,
                        'original_count': len(data),
                        'descriptor_length': 128,
                        'data_type': 'sift_descriptors',
                        'total_elements': len(flat_data)
                    }
                
                # For other nested arrays, try to flatten
                else:
                    print(f"Flattening generic nested array at {path}")
                    try:
                        flat_array = []
                        for sublist in data:
                            if isinstance(sublist, list):
                                flat_array.extend(sublist)
                            else:
                                flat_array.append(sublist)
                        return flat_array
                    except:
                        # If flattening fails, convert to string representation
                        return f"[Nested array with {len(data)} items - flattened to string]"
            else:
                # Regular list, just clean the items
                return [flatten_nested_arrays(item, f"{path}[{i}]") for i in range(min(len(data), 1000))]
        
        else:
            return data
            
    except Exception as e:
        print(f"Error flattening arrays at {path}: {e}")
        return f"[Error: {str(e)[:100]}]"


def create_firestore_safe_features(animal_data):
    """
    Create a Firestore-safe version of the features by handling problematic nested arrays
    *** UPDATED TO PRESERVE DOGFACENET EMBEDDINGS ***
    """
    try:
        safe_data = []
        
        for animal in animal_data:
            safe_animal = {}
            
            # Copy basic fields
            for key in ['animal_type', 'confidence', 'bounding_box', 'center_point']:
                if key in animal:
                    safe_animal[key] = animal[key]
            
            # Handle muzzle_features specially
            if 'muzzle_features' in animal:
                mf = animal['muzzle_features']
                safe_mf = {}
                
                # Handle traditional features
                if 'traditional_features' in mf and mf['traditional_features']:
                    tf = mf['traditional_features']
                    safe_tf = {}
                    
                    # Copy metadata
                    for key in ['feature_dimension', 'texture_dim', 'multi_scale_dim']:
                        if key in tf:
                            safe_tf[key] = tf[key]
                    
                    # Handle the features array
                    if 'features' in tf:
                        features = tf['features']
                        if len(features) > 2000:  # Limit size
                            safe_tf['features'] = features[:2000]
                            safe_tf['features_truncated'] = True
                            safe_tf['original_feature_count'] = len(features)
                        else:
                            safe_tf['features'] = features
                    
                    # Handle muzzle_size
                    if 'muzzle_size' in tf:
                        safe_tf['muzzle_size'] = list(tf['muzzle_size']) if tf['muzzle_size'] else [0, 0]
                    
                    safe_mf['traditional_features'] = safe_tf
                
                # *** NEW: Handle DogFaceNet embeddings ***
                if 'dogfacenet_embeddings' in mf and mf['dogfacenet_embeddings']:
                    embeddings = mf['dogfacenet_embeddings']
                    print(f"Preserving DogFaceNet embeddings: {len(embeddings)} dimensions")
                    
                    # DogFaceNet embeddings are already flat arrays, so they're safe for Firestore
                    safe_mf['dogfacenet_embeddings'] = embeddings
                    safe_mf['dogfacenet_size'] = len(embeddings)
                    safe_mf['has_dogfacenet'] = True
                else:
                    safe_mf['has_dogfacenet'] = False
                
                # Handle SIFT features (existing code)
                if 'sift_features' in mf and mf['sift_features']:
                    sf = mf['sift_features']
                    safe_sf = {}
                    
                    # Copy metadata
                    for key in ['keypoints_count', 'descriptor_dimension', 'feature_count']:
                        if key in sf:
                            safe_sf[key] = sf[key]
                    
                    # Handle descriptors - flatten them
                    if 'descriptors' in sf:
                        descriptors = sf['descriptors']
                        print(f"Processing SIFT descriptors: {len(descriptors)} descriptors")
                        
                        flattened_descriptors = []
                        for desc in descriptors:
                            if isinstance(desc, list):
                                flattened_descriptors.extend(desc)
                            else:
                                flattened_descriptors.append(desc)
                        
                        safe_sf['descriptors_flat'] = flattened_descriptors
                        safe_sf['descriptors_count'] = len(descriptors)
                        safe_sf['descriptor_length'] = len(descriptors[0]) if descriptors else 0
                        safe_sf['storage_format'] = 'flattened'
                        
                        print(f"SIFT descriptors flattened: {len(flattened_descriptors)} total elements")
                    
                    safe_mf['sift_features'] = safe_sf
                
                # *** NEW: Copy other feature metadata ***
                for key in ['has_both', 'feature_types', 'has_sift', 'embedding_size']:
                    if key in mf:
                        safe_mf[key] = mf[key]
                
                safe_animal['muzzle_features'] = safe_mf
            
            safe_data.append(safe_animal)
        
        return safe_data
        
    except Exception as e:
        print(f"Error creating safe features: {e}")
        import traceback
        traceback.print_exc()
        return animal_data


def reconstruct_sift_descriptors(safe_sift_features):
    """
    Reconstruct SIFT descriptors from flattened format for similarity calculations
    """
    try:
        if 'descriptors_flat' not in safe_sift_features:
            return None
        
        flat_descriptors = safe_sift_features['descriptors_flat']
        descriptor_count = safe_sift_features.get('descriptors_count', 0)
        descriptor_length = safe_sift_features.get('descriptor_length', 128)
        
        if len(flat_descriptors) != descriptor_count * descriptor_length:
            print("Warning: Flattened descriptor size doesn't match expected dimensions")
            return None
        
        # Reconstruct nested structure
        reconstructed = []
        for i in range(descriptor_count):
            start_idx = i * descriptor_length
            end_idx = start_idx + descriptor_length
            descriptor = flat_descriptors[start_idx:end_idx]
            reconstructed.append(descriptor)
        
        return {
            'keypoints_count': safe_sift_features.get('keypoints_count'),
            'descriptors': reconstructed,
            'descriptor_dimension': descriptor_length,
            'feature_count': descriptor_count
        }
        
    except Exception as e:
        print(f"Error reconstructing SIFT descriptors: {e}")
        return None


def store_pet_with_safe_arrays(db, snout_id, pet_name, pet_breed, owner_name, animal_data, file_path):
    """
    Store pet with specifically safe array handling
    """
    try:
        print("\n" + "="*60)
        print("STORING PET WITH SAFE ARRAY HANDLING")
        print("="*60)
        
        # Create Firestore-safe version of the data
        print("Creating Firestore-safe features...")
        safe_animal_data = create_firestore_safe_features(animal_data)
        
        # Create the document
        pet_doc = {
            'snout_id': snout_id,
            'name': pet_name,
            'breed': pet_breed,
            'owner': owner_name,
            'muzzle_data': safe_animal_data,
            'image_path': file_path,
            'created_at': firestore.SERVER_TIMESTAMP,
            'storage_format': 'safe_arrays',
            'animals_detected': len(safe_animal_data)
        }
        
        # Test document size
        import json
        doc_size = len(json.dumps(pet_doc, default=str))
        print(f"Safe document size: {doc_size} bytes ({doc_size/1024:.1f} KB)")
        
        # Store directly (no minimal document strategy)
        print("Storing complete document...")
        db.collection('my_pets').document(snout_id).set(pet_doc)
        
        print(f"Pet data successfully stored with safe array format!")
        return True
        
    except Exception as e:
        print(f"Safe array storage failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def calculate_sift_similarity_safe(features1, features2, threshold=0.75):
    """
    Calculate SIFT similarity with support for flattened descriptors
    """
    try:
        if not features1 or not features2:
            return 0.0, False
        
        # Check if we need to reconstruct descriptors
        sift1 = features1.get('sift_features')
        sift2 = features2.get('sift_features')
        
        if not sift1 or not sift2:
            return 0.0, False
        
        # Reconstruct if flattened
        if 'descriptors_flat' in sift1:
            sift1 = reconstruct_sift_descriptors(sift1)
        if 'descriptors_flat' in sift2:
            sift2 = reconstruct_sift_descriptors(sift2)
        
        if not sift1 or not sift2:
            print("Failed to reconstruct SIFT descriptors")
            return 0.0, False
        
        # Now use the original SIFT similarity calculation
        return sift_enhancer.calculate_sift_similarity(sift1, sift2, threshold)
        
    except Exception as e:
        print(f"Error in safe SIFT similarity: {e}")
        return 0.0, False


def calculate_flattened_similarity(features1, features2, threshold=0.75):
    """
    Calculate similarity between flattened features (as stored in database)
    """
    try:
        print(f"\n   🔧 FLATTENED SIMILARITY CALCULATION:")
        print(f"   " + "-" * 50)
        
        if not features1 or not features2:
            print("   ❌ One or both feature sets are empty")
            return 0.0, False
        
        print(f"   ✅ Both feature sets present")
        print(f"   Features1 type: {type(features1)}")
        print(f"   Features2 type: {type(features2)}")
        
        # Handle traditional features comparison
        traditional_sim = 0.0
        traditional_match = False
        
        if ('traditional_features' in features1 and 'traditional_features' in features2 and
            features1['traditional_features'] and features2['traditional_features']):
            print("   🔍 Comparing traditional features...")
            
            tf1 = features1['traditional_features']
            tf2 = features2['traditional_features']
            
            if 'features' in tf1 and 'features' in tf2:
                print(f"   Traditional features length: {len(tf1['features'])} vs {len(tf2['features'])}")
                traditional_sim, traditional_match = calculate_improved_similarity(tf1, tf2, threshold)
                print(f"   Traditional similarity: {traditional_sim:.4f}, match: {traditional_match}")
        else:
            print("   ❌ Missing traditional features in one or both sets")
        
        # Handle SIFT features comparison with flattened descriptors
        sift_sim = 0.0
        sift_match = False
        
        if ('sift_features' in features1 and 'sift_features' in features2 and
            features1['sift_features'] and features2['sift_features']):
            print("   🔍 Comparing SIFT features...")
            
            sf1 = features1['sift_features']
            sf2 = features2['sift_features']
            
            # Check if we have flattened descriptors
            if ('descriptors_flat' in sf1 and 'descriptors_flat' in sf2):
                print("   🔧 Reconstructing flattened SIFT descriptors...")
                
                # Reconstruct SIFT descriptors from flattened format
                reconstructed_sf1 = reconstruct_sift_descriptors(sf1)
                reconstructed_sf2 = reconstruct_sift_descriptors(sf2)
                
                if reconstructed_sf1 and reconstructed_sf2:
                    print("   ✅ SIFT descriptors reconstructed successfully")
                    sift_sim, sift_match = sift_enhancer.calculate_sift_similarity(
                        reconstructed_sf1, reconstructed_sf2, threshold
                    )
                    print(f"   SIFT similarity: {sift_sim:.4f}, match: {sift_match}")
                else:
                    print("   ❌ Failed to reconstruct SIFT descriptors")
            elif ('descriptors' in sf1 and 'descriptors' in sf2):
                print("   🔍 Using regular SIFT descriptors...")
                sift_sim, sift_match = sift_enhancer.calculate_sift_similarity(sf1, sf2, threshold)
                print(f"   SIFT similarity: {sift_sim:.4f}, match: {sift_match}")
            else:
                print("   ❌ No valid SIFT descriptors found")
        else:
            print("   ❌ Missing SIFT features in one or both sets")
        
        # Combine similarities
        if traditional_sim > 0 and sift_sim > 0:
            # Both traditional and SIFT features available
            combined_similarity = 0.6 * sift_sim + 0.4 * traditional_sim
            combined_match = combined_similarity >= threshold
            print(f"   🎯 Combined similarity: SIFT={sift_sim:.3f}, Traditional={traditional_sim:.3f}, Combined={combined_similarity:.3f}")
        elif sift_sim > 0:
            # Only SIFT available
            combined_similarity = sift_sim
            combined_match = sift_match
            print(f"   🎯 SIFT-only similarity: {sift_sim:.3f}")
        elif traditional_sim > 0:
            # Only traditional features
            combined_similarity = traditional_sim
            combined_match = traditional_match
            print(f"   🎯 Traditional-only similarity: {traditional_sim:.3f}")
        else:
            # No valid features
            combined_similarity = 0.0
            combined_match = False
            print(f"   ❌ No valid features for comparison")
        
        return combined_similarity, combined_match
        
    except Exception as e:
        print(f"   ❌ Error in flattened similarity calculation: {e}")
        import traceback
        traceback.print_exc()
        return 0.0, False


def calculate_flattened_similarity_with_dogfacenet(features1, features2, threshold=0.75):
    """
    Calculate similarity between flattened features INCLUDING DogFaceNet embeddings
    *** UPDATED TO USE DOGFACENET ***
    """
    try:
        print(f"\n   🔍 ENHANCED SIMILARITY WITH DOGFACENET:")
        print(f"   " + "-" * 50)
        
        if not features1 or not features2:
            print("   ❌ One or both feature sets are empty")
            return 0.0, False
        
        print(f"   ✅ Both feature sets present")
        
        similarities = {}
        weights = {}
        
        # 1. Traditional features comparison
        if ('traditional_features' in features1 and 'traditional_features' in features2 and
            features1['traditional_features'] and features2['traditional_features']):
            print("   🔍 Comparing traditional features...")
            
            tf1 = features1['traditional_features']
            tf2 = features2['traditional_features']
            
            if 'features' in tf1 and 'features' in tf2:
                print(f"   Traditional features length: {len(tf1['features'])} vs {len(tf2['features'])}")
                traditional_sim, traditional_match = calculate_improved_similarity(tf1, tf2, threshold)
                similarities['traditional'] = traditional_sim
                weights['traditional'] = 0.25
                print(f"   Traditional similarity: {traditional_sim:.4f}")
        
        # 2. *** NEW: DogFaceNet embeddings comparison ***
        if ('dogfacenet_embeddings' in features1 and 'dogfacenet_embeddings' in features2 and
            features1['dogfacenet_embeddings'] and features2['dogfacenet_embeddings']):
            print("   🐕 Comparing DogFaceNet embeddings...")
            
            embeddings1 = features1['dogfacenet_embeddings']
            embeddings2 = features2['dogfacenet_embeddings']
            
            print(f"   DogFaceNet embeddings: {len(embeddings1)} vs {len(embeddings2)} dimensions")
            
            # Calculate cosine similarity for embeddings (standard for face recognition)
            if len(embeddings1) == len(embeddings2):
                import numpy as np
                
                emb1 = np.array(embeddings1)
                emb2 = np.array(embeddings2)
                
                # Cosine similarity (embeddings should already be L2 normalized)
                dogfacenet_sim = np.dot(emb1, emb2)
                # Convert to [0,1] range
                dogfacenet_sim = (dogfacenet_sim + 1) / 2
                
                similarities['dogfacenet'] = dogfacenet_sim
                weights['dogfacenet'] = 0.45  # Highest weight for specialized model
                print(f"   🐕 DogFaceNet similarity: {dogfacenet_sim:.4f}")
            else:
                print("   ❌ DogFaceNet embedding dimension mismatch")
        else:
            print("   ❌ Missing DogFaceNet embeddings in one or both sets")
        
        # 3. SIFT features comparison (existing)
        if ('sift_features' in features1 and 'sift_features' in features2 and
            features1['sift_features'] and features2['sift_features']):
            print("   🔍 Comparing SIFT features...")
            
            sf1 = features1['sift_features']
            sf2 = features2['sift_features']
            
            if ('descriptors_flat' in sf1 and 'descriptors_flat' in sf2):
                print("   🔧 Reconstructing flattened SIFT descriptors...")
                
                reconstructed_sf1 = reconstruct_sift_descriptors(sf1)
                reconstructed_sf2 = reconstruct_sift_descriptors(sf2)
                
                if reconstructed_sf1 and reconstructed_sf2:
                    print("   ✅ SIFT descriptors reconstructed successfully")
                    sift_sim, sift_match = sift_enhancer.calculate_sift_similarity(
                        reconstructed_sf1, reconstructed_sf2, threshold
                    )
                    similarities['sift'] = sift_sim
                    weights['sift'] = 0.30
                    print(f"   SIFT similarity: {sift_sim:.4f}")
                else:
                    print("   ❌ Failed to reconstruct SIFT descriptors")
        
        # Calculate weighted combination
        if similarities:
            # Normalize weights
            total_weight = sum(weights[k] for k in similarities.keys())
            
            combined_similarity = sum(
                similarities[method] * (weights[method] / total_weight)
                for method in similarities.keys()
            )
            
            # Adjust threshold based on available methods
            adjusted_threshold = threshold
            if 'dogfacenet' in similarities:
                adjusted_threshold = threshold - 0.05  # More lenient with DogFaceNet
            
            is_match = combined_similarity >= adjusted_threshold
            
            print(f"   🎯 Combined similarity: {combined_similarity:.4f}")
            print(f"   Methods used: {list(similarities.keys())}")
            print(f"   Weights used: {[(k, f'{weights[k]/total_weight:.2f}') for k in similarities.keys()]}")
            print(f"   Match: {'✅' if is_match else '❌'} (threshold: {adjusted_threshold:.2f})")
            
            return combined_similarity, is_match
        else:
            print("   ❌ No valid features for comparison")
            return 0.0, False
            
    except Exception as e:
        print(f"   ❌ Error in DogFaceNet similarity calculation: {e}")
        import traceback
        traceback.print_exc()
        return 0.0, False


def detect_animals_and_extract_features(image):
    """Improved animal detection with muzzle feature extraction"""
    try:
        # Check if model is available
        if model is None:
            print("YOLOv5 model not available")
            return []
        
        # Apply enhanced preprocessing for better detection
        enhanced_image = preprocess_image_for_better_detection(image)
        print(f"Enhanced image preprocessing completed, shape: {enhanced_image.shape}")
        
        # Debug: Check if preprocessing changed the image significantly
        print(f"Original image range: [{image.min()}, {image.max()}]")
        print(f"Enhanced image range: [{enhanced_image.min()}, {enhanced_image.max()}]")
        
        print(f"Starting YOLOv5 detection on enhanced image shape: {enhanced_image.shape}")
        
        # Perform detection using YOLOv5 on enhanced image
        print(f"Running YOLOv5 inference on image of shape {enhanced_image.shape}...")
        results = model(enhanced_image)
        
        print(f"YOLOv5 results structure: {type(results)}")
        print(f"YOLOv5 results attributes: {dir(results)}")
        print(f"YOLOv5 xyxy length: {len(results.xyxy)}")
        
        # Get predictions - use .xyxy for absolute coordinates
        predictions = results.xyxy[0].cpu().numpy()  # x1, y1, x2, y2, confidence, class
        
        print(f"YOLOv5 detected {len(predictions)} objects")
        
        # Debug: Show all detections regardless of confidence
        if len(predictions) > 0:
            print("All detections (including low confidence):")
            for i, detection in enumerate(predictions):
                x1, y1, x2, y2, confidence, class_id = detection
                class_id = int(class_id)
                class_name = results.names[class_id]
                print(f"  Detection {i}: {class_name} (ID: {class_id}) with confidence {confidence:.3f}")
        
        if len(predictions) == 0:
            print("No objects detected by YOLOv5 - this might indicate an issue with the model or image")
            print("Available YOLOv5 classes: ", list(results.names.values()))
            
            # Fallback: Try to detect animals using face detection and basic analysis
            print("Attempting fallback animal detection...")
            fallback_animals = detect_animals_fallback(enhanced_image)
            if fallback_animals:
                print(f"Fallback detection found {len(fallback_animals)} animals")
                return fallback_animals
            else:
                print("Fallback detection also failed")
            return []

        img_height, img_width, _ = image.shape
        animal_data = []

        # YOLOv5 COCO class names for animals
        ANIMAL_CLASS_IDS = {
            15: 'bird', 16: 'cat', 17: 'dog', 18: 'horse', 19: 'sheep', 
            20: 'cow', 21: 'elephant', 22: 'bear', 23: 'zebra', 24: 'giraffe'
        }
        
        print(f"Available YOLOv5 classes: {list(results.names.values())}")
        print(f"Looking for animal class IDs: {list(ANIMAL_CLASS_IDS.keys())}")

        for i, detection in enumerate(predictions):
            x1, y1, x2, y2, confidence, class_id = detection
            class_id = int(class_id)
            class_name = results.names[class_id]
            
            print(f"Processing detection {i}: {class_name} (ID: {class_id}) with confidence {confidence:.3f}")
            print(f"Bounding box: x1={x1:.1f}, y1={y1:.1f}, x2={x2:.1f}, y2={y2:.1f}")

            # Check if it's an animal and meets confidence threshold
            is_animal = class_id in ANIMAL_CLASS_IDS or class_name.lower() in ['cat', 'dog', 'horse', 'cow', 'sheep', 'bear', 'elephant']
            
            if is_animal and confidence > 0.1:  # Lower confidence threshold for better detection
                print(f"Animal detected: {class_name} (ID: {class_id})")
                
                # Convert coordinates to integers and ensure they're within image bounds
                left = max(0, int(x1))
                top = max(0, int(y1))
                right = min(img_width, int(x2))
                bottom = min(img_height, int(y2))
                
                # Validate bounding box
                if left >= right or top >= bottom:
                    print(f"Invalid bounding box after conversion: left={left}, top={top}, right={right}, bottom={bottom}")
                    continue
                
                # Check if bounding box is reasonable size
                box_width = right - left
                box_height = bottom - top
                
                if box_width < 20 or box_height < 20:
                    print(f"Bounding box too small: width={box_width}, height={box_height}")
                    continue
                
                print(f"Valid bounding box: left={left}, top={top}, right={right}, bottom={bottom}")
                print(f"Box dimensions: width={box_width}, height={box_height}")

                # Extract comprehensive muzzle features with machine learning
                print(f"Extracting features for {class_name}...")
                muzzle_features = extract_super_enhanced_muzzle_features(
                    enhanced_image, (left, top, right, bottom)
                )

                if muzzle_features is not None:
                    print(f"Features extracted successfully for {class_name}")
                    animal_info = {
                        'animal_type': class_name,
                        'confidence': float(confidence),
                        'bounding_box': {
                            'left': left, 'top': top,
                            'right': right, 'bottom': bottom,
                            'width': box_width, 'height': box_height
                        },
                        'muzzle_features': muzzle_features,
                        'center_point': {
                            'x': int((left + right) / 2),
                            'y': int((top + bottom) / 2)
                        }
                    }
                    animal_data.append(animal_info)
                    
                    # Validate and debug the extracted features
                    validate_and_debug_features([animal_info], class_name)
                else:
                    print(f"Failed to extract features for {class_name}")
            else:
                if not is_animal:
                    print(f"Skipping {class_name} (not classified as animal)")
                else:
                    print(f"Skipping {class_name} (confidence {confidence:.3f} too low)")

        print(f"Total animals processed: {len(animal_data)}")
        return animal_data
        
    except Exception as e:
        print(f"Error in detect_animals_and_extract_features: {e}")
        import traceback
        traceback.print_exc()
        return []


def detect_animals_fallback(image):
    """Fallback animal detection using face detection and basic image analysis"""
    try:
        print("Starting fallback animal detection...")
        
        # Convert to grayscale for face detection
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image
        
        # Use dlib face detector (works well for animal faces too) or OpenCV fallback
        if DLIB_AVAILABLE:
            detector = dlib.get_frontal_face_detector()
            faces = detector(gray_image, 1)  # Upsample once for better detection
        else:
            # Fallback to OpenCV face detector
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces_cv = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5)
            # Convert OpenCV format to dlib-like format for compatibility
            faces = [dlib.rectangle(int(x), int(y), int(x+w), int(y+h)) if DLIB_AVAILABLE 
                    else (x, y, x+w, y+h) for (x, y, w, h) in faces_cv]
        
        print(f"Fallback detection found {len(faces)} potential animal faces")
        
        animal_data = []
        img_height, img_width = gray_image.shape
        
        for i, face in enumerate(faces):
            # Get bounding box coordinates
            left = max(0, face.left())
            top = max(0, face.top())
            right = min(img_width, face.right())
            bottom = min(img_height, face.bottom())
            
            # Validate bounding box
            if left >= right or top >= bottom:
                continue
                
            box_width = right - left
            box_height = bottom - top
            
            # Check if bounding box is reasonable size
            if box_width < 50 or box_height < 50:  # Larger minimum for fallback
                continue
            
            print(f"Fallback animal {i+1}: face detected at ({left}, {top}, {right}, {bottom})")
            print(f"Box dimensions: width={box_width}, height={box_height}")
            
            # Extract features using the enhanced method with machine learning
            print(f"Extracting features for fallback animal {i+1}...")
            muzzle_features = extract_super_enhanced_muzzle_features(
                image, (left, top, right, bottom)
            )
            
            if muzzle_features is not None:
                print(f"Features extracted successfully for fallback animal {i+1}")
                animal_info = {
                    'animal_type': 'unknown_animal',  # We don't know the specific type
                    'confidence': 0.5,  # Default confidence for fallback
                    'bounding_box': {
                        'left': left, 'top': top,
                        'right': right, 'bottom': bottom,
                        'width': box_width, 'height': box_height
                    },
                    'muzzle_features': muzzle_features,
                    'center_point': {
                        'x': int((left + right) / 2),
                        'y': int((top + bottom) / 2)
                    },
                    'detection_method': 'fallback_face_detection'
                }
                animal_data.append(animal_info)
                
                # Validate and debug the extracted features
                validate_and_debug_features([animal_info], f"fallback_animal_{i+1}")
            else:
                print(f"Failed to extract features for fallback animal {i+1}")
        
        return animal_data

    except Exception as e:
        print(f"Error in fallback animal detection: {e}")
        import traceback
        traceback.print_exc()
        return []


def calculate_feature_similarity(features1, features2, threshold=0.75):
    """Calculate similarity between two feature vectors"""
    # Use the enhanced similarity calculation with SIFT and traditional features
    return calculate_enhanced_similarity(features1, features2, threshold)


def process_image(file_path):
    """Common image processing function"""
    try:
        print(f"Processing image: {file_path}")
        print(f"File exists: {os.path.exists(file_path)}")
        print(f"File size: {os.path.getsize(file_path) if os.path.exists(file_path) else 'N/A'} bytes")
        
        # Try direct OpenCV read first (most reliable)
        cv_image = cv2.imread(file_path)
        if cv_image is not None:
            print(f"Direct OpenCV read successful, shape: {cv_image.shape}")
            return cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        print("Direct OpenCV read failed, trying PIL approach...")
        
        # Fallback to PIL if OpenCV fails
        try:
            image = Image.open(file_path)
            print(f"PIL image opened successfully, size: {image.size}")
            
            # Convert PIL to OpenCV format
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert PIL to numpy array
            img_array = np.array(image)
            print(f"Converted to numpy array, shape: {img_array.shape}")
            
            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            print(f"Converted to BGR, shape: {img_bgr.shape}")
            
            # Now convert back to RGB for our processing
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            print(f"Final RGB image, shape: {img_rgb.shape}")
            
            return img_rgb
            
        except Exception as pil_error:
            print(f"PIL processing failed: {pil_error}")
            
            # Last resort: try to read the file as bytes and decode
            try:
                with open(file_path, 'rb') as f:
                    file_bytes = f.read()
                    print(f"Read {len(file_bytes)} bytes from file")
                    
                    # Try to decode with OpenCV from memory
                    nparr = np.frombuffer(file_bytes, np.uint8)
                    cv_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if cv_image is not None:
                        print(f"OpenCV decode from bytes successful, shape: {cv_image.shape}")
                        return cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
                    else:
                        print("OpenCV decode from bytes failed")
                        
            except Exception as bytes_error:
                print(f"Bytes processing failed: {bytes_error}")
        
        print("All image processing methods failed")
        return None
        
    except Exception as e:
        print(f"Error in process_image: {e}")
        return None


@app.route('/scanFace', methods=['POST'])
def scan_face():
    try:
        if 'file' not in request.files:
            return jsonify({"message": "No file part"}), 400

        uploaded_file = request.files['file']
        if uploaded_file.filename == '':
            return jsonify({"message": "No selected file"}), 400

        file_path = os.path.join('uploads', 'scan.jpg')
        uploaded_file.save(file_path)

        # Process image
        image_rgb = process_image(file_path)
        if image_rgb is None:
            return jsonify({"message": "Error processing the image"}), 500

        # Extract features with enhanced preprocessing and SIFT
        animal_data = detect_animals_and_extract_features(image_rgb)
        
        # If no animals detected and YOLOv5 is not available, try fallback detection
        if not animal_data and model is None:
            print("YOLOv5 model not available, trying fallback detection...")
            animal_data = detect_animals_fallback(image_rgb)
        
        if not animal_data:
            return jsonify({"message": "No animal muzzle detected"}), 400

        return jsonify({
            "message": "Animal features extracted successfully",
            "animals_detected": len(animal_data),
            "animal_data": animal_data
        }), 200

    except Exception as e:
        return jsonify({"message": f"Error: {e}"}), 500


@app.route('/storeSnout', methods=['POST'])
def store_snout():
    try:
        print("=== /storeSnout endpoint called ===")
        
        if 'file' not in request.files:
            return jsonify({"message": "No file part"}), 400

        uploaded_file = request.files['file']
        if uploaded_file.filename == '':
            return jsonify({"message": "No selected file"}), 400

        # Get additional pet info
        pet_name = request.form.get('pet_name', 'Unknown')
        pet_breed = request.form.get('pet_breed', 'Unknown')
        owner_name = request.form.get('owner_name', 'Unknown')

        print(f"Storing pet: {pet_name}, breed: {pet_breed}, owner: {owner_name}")

        snout_id = str(uuid.uuid4())
        file_path = os.path.join('snout_data', f'{snout_id}.jpg')
        uploaded_file.save(file_path)
        
        print(f"File saved with snout_id: {snout_id}")

        # Process image
        image_rgb = process_image(file_path)
        if image_rgb is None:
            return jsonify({"message": "Error processing the image"}), 500

        print(f"Image processed successfully, shape: {image_rgb.shape}")

        # Extract features
        animal_data = detect_animals_and_extract_features(image_rgb)
        if not animal_data:
            return jsonify({"message": "No animal muzzle detected"}), 400

        print(f"Features extracted for {len(animal_data)} animals")

        # Debug: Check DogFaceNet preservation after extraction
        debug_dogfacenet_preservation(animal_data, "after_extraction")

        # Store in Firebase if available
        storage_success = False
        if db is not None:
            print("Database available, attempting safe array storage...")
            storage_success = store_pet_with_safe_arrays(
                db, snout_id, pet_name, pet_breed, owner_name, animal_data, file_path
            )
        else:
            print("Database not available")

        # Prepare response
        response_data = {
            "message": "Pet muzzle data processed successfully",
            "snout_id": snout_id,
            "pet_name": pet_name,
            "animals_detected": len(animal_data),
            "storage_success": storage_success,
            "storage_format": "safe_arrays",
            "muzzle_data": animal_data  # Include the actual muzzle features
        }
        
        # Only include debugging info if storage failed
        if not storage_success:
            response_data["note"] = "Storage failed - check server logs for details"
            response_data["debug_info"] = {
                "animals_detected": len(animal_data),
                "has_muzzle_features": bool(animal_data[0].get('muzzle_features')) if animal_data else False,
                "database_available": db is not None
            }

        return jsonify(response_data), 200

    except Exception as e:
        print(f"Error in store_snout: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"message": f"Error: {e}"}), 500


@app.route('/identifyPet', methods=['POST'])
def identify_pet():
    try:
        print("\n" + "="*80)
        print("🔍 /identifyPet endpoint called")
        print("="*80)
        
        # Log all request data
        print("\n📋 REQUEST DATA ANALYSIS:")
        print("-" * 50)
        print(f"Request method: {request.method}")
        print(f"Request headers: {dict(request.headers)}")
        print(f"Request form data: {dict(request.form)}")
        print(f"Request files: {list(request.files.keys())}")
        
        # Detailed form data analysis
        if request.form:
            print(f"\n📝 FORM DATA DETAILS:")
            print("-" * 30)
            for key, value in request.form.items():
                print(f"   {key}: {value}")
        else:
            print(f"\n📝 No form data received")
        
        # Detailed file analysis
        if request.files:
            print(f"\n📁 FILE DATA DETAILS:")
            print("-" * 30)
            for key, file in request.files.items():
                print(f"   File key: {key}")
                print(f"   Filename: {file.filename}")
                print(f"   Content type: {file.content_type}")
                if hasattr(file, 'content_length'):
                    print(f"   Content length: {file.content_length}")
                else:
                    print(f"   Content length: Unknown")
        else:
            print(f"\n📁 No files received")
        
        if 'file' not in request.files:
            print("❌ No file part in request")
            return jsonify({"message": "No file part"}), 400

        uploaded_file = request.files['file']
        if uploaded_file.filename == '':
            print("❌ No selected file")
            return jsonify({"message": "No selected file"}), 400

        print(f"\n📁 FILE DETAILS:")
        print("-" * 30)
        print(f"Filename: {uploaded_file.filename}")
        print(f"Content type: {uploaded_file.content_type}")
        print(f"Content length: {uploaded_file.content_length if hasattr(uploaded_file, 'content_length') else 'Unknown'}")
        
        file_path = os.path.join('uploads', 'identify.jpg')
        uploaded_file.save(file_path)
        print(f"File saved to: {file_path}")
        print(f"Saved file size: {os.path.getsize(file_path)} bytes")

        # Process image
        print("\n🖼️ IMAGE PROCESSING:")
        print("-" * 30)
        print("Starting image processing...")
        image_rgb = process_image(file_path)
        if image_rgb is None:
            print("❌ Image processing failed")
            return jsonify({"message": "Error processing the image"}), 500

        print(f"✅ Image processing successful")
        print(f"   Image shape: {image_rgb.shape}")
        print(f"   Image dtype: {image_rgb.dtype}")
        print(f"   Image range: [{image_rgb.min():.2f}, {image_rgb.max():.2f}]")

        # Extract features from query image with enhanced preprocessing and SIFT
        print("\n🔍 FEATURE EXTRACTION:")
        print("-" * 30)
        print("Starting animal detection and feature extraction...")
        raw_query_animal_data = detect_animals_and_extract_features(image_rgb)
        if not raw_query_animal_data:
            print("❌ No animals detected")
            return jsonify({"message": "No animal muzzle detected"}), 400

        print(f"✅ Raw features extracted for {len(raw_query_animal_data)} animals")
        
        # Convert to the same format used in database storage
        print("\n🔄 CONVERTING TO DATABASE FORMAT:")
        print("-" * 40)
        print("Converting features to match stored format...")
        
        # Debug: Show original format
        print(f"Original features structure:")
        for i, animal in enumerate(raw_query_animal_data):
            if 'muzzle_features' in animal:
                mf = animal['muzzle_features']
                print(f"  Animal {i+1}: {list(mf.keys())}")
                if 'sift_features' in mf and mf['sift_features']:
                    sf = mf['sift_features']
                    if 'descriptors' in sf:
                        desc = sf['descriptors']
                        print(f"    SIFT descriptors: {len(desc)} descriptors")
                        if len(desc) > 0:
                            print(f"    First descriptor length: {len(desc[0])}")
        
        query_animal_data = create_firestore_safe_features(raw_query_animal_data)
        
        if not query_animal_data:
            print("❌ Failed to convert features to database format")
            return jsonify({"message": "Failed to process features"}), 500
        
        # Debug: Check DogFaceNet preservation after safe conversion
        debug_dogfacenet_preservation(query_animal_data, "after_safe_conversion")
        
        # Debug: Show converted format
        print(f"\nConverted features structure:")
        for i, animal in enumerate(query_animal_data):
            if 'muzzle_features' in animal:
                mf = animal['muzzle_features']
                print(f"  Animal {i+1}: {list(mf.keys())}")
                if 'sift_features' in mf and mf['sift_features']:
                    sf = mf['sift_features']
                    if 'descriptors_flat' in sf:
                        flat_desc = sf['descriptors_flat']
                        print(f"    Flattened SIFT: {len(flat_desc)} values")
                        print(f"    Descriptor count: {sf.get('descriptor_count', 'N/A')}")
                        print(f"    Descriptor length: {sf.get('descriptor_length', 'N/A')}")

        print(f"✅ Animals detected: {len(query_animal_data)}")
        
        # Debug: Show what we detected
        print("\n📊 EXTRACTED FEATURES ANALYSIS:")
        print("-" * 40)
        for i, animal in enumerate(query_animal_data):
            print(f"\n🐾 Query Animal {i+1}:")
            print(f"   Animal type: {animal['animal_type']}")
            print(f"   Confidence: {animal['confidence']:.3f}")
            print(f"   Bounding box: {animal.get('bounding_box', 'N/A')}")
            print(f"   Center point: {animal.get('center_point', 'N/A')}")
            
            if 'muzzle_features' in animal:
                muzzle_features = animal['muzzle_features']
                print(f"   ✅ Has muzzle features")
                print(f"   Feature keys: {list(muzzle_features.keys())}")
                
                # Log traditional features details
                if 'traditional_features' in muzzle_features:
                    trad_features = muzzle_features['traditional_features']
                    if 'features' in trad_features:
                        features = trad_features['features']
                        print(f"   Traditional features: {len(features)} values")
                        print(f"   Feature range: [{min(features):.4f}, {max(features):.4f}]")
                        print(f"   First 5 values: {features[:5]}")
                        print(f"   Last 5 values: {features[-5:]}")
                
                # Log SIFT features details
                if 'sift_features' in muzzle_features:
                    sift_features = muzzle_features['sift_features']
                    print(f"   SIFT features: {len(sift_features)} keypoints")
                    
                    # Check if we have flattened descriptors (converted format)
                    if 'descriptors_flat' in sift_features:
                        flat_desc = sift_features['descriptors_flat']
                        print(f"   ✅ Flattened descriptors: {len(flat_desc)} values")
                        print(f"   Descriptor count: {sift_features.get('descriptor_count', 'N/A')}")
                        print(f"   Descriptor length: {sift_features.get('descriptor_length', 'N/A')}")
                    elif 'descriptors' in sift_features:
                        descriptors = sift_features['descriptors']
                        if hasattr(descriptors, 'shape'):
                            print(f"   Descriptors shape: {descriptors.shape}")
                        else:
                            print(f"   Descriptors type: {type(descriptors)}")
                            print(f"   Descriptors length: {len(descriptors) if hasattr(descriptors, '__len__') else 'N/A'}")
            else:
                print(f"   ❌ No muzzle features found")

        # Compare with stored pets
        print("\n🗄️ DATABASE SEARCH:")
        print("-" * 30)
        
        if db is None:
            print("❌ Database not available for comparison - returning detection results only")
            # Return successful detection without database comparison
            return jsonify({
                "message": "Pet detection completed successfully",
                "query_animals_detected": len(query_animal_data),
                "matches_found": 0,
                "matches": [],
                "note": "Database unavailable - showing detection results only"
            }), 200

        print("✅ Database available, searching for matches...")
        matches = []
        try:
            # Look in the correct collection where Flutter stores pets
            print(f"🔍 Searching in collection: 'my_pets'")
            pets_ref = db.collection('my_pets')
            stored_pets = pets_ref.stream()
            
            pet_count = 0
            print("\n📋 STORED PETS ANALYSIS:")
            print("-" * 40)
            
            for stored_pet_doc in stored_pets:
                pet_count += 1
                stored_pet = stored_pet_doc.to_dict()
                print(f"\n🐕 Pet {pet_count}: {stored_pet.get('name', 'Unknown')}")
                print(f"   Pet ID: {stored_pet_doc.id}")
                print(f"   Animal type: {stored_pet.get('animal_type', 'Not set')}")
                print(f"   Breed: {stored_pet.get('breed', 'Unknown')}")
                print(f"   Owner: {stored_pet.get('owner_name', stored_pet.get('owner', 'Unknown'))}")
                
                # Check muzzle features
                stored_muzzle_features = stored_pet.get('muzzle_features', [])
                print(f"   Muzzle features type: {type(stored_muzzle_features)}")
                
                if isinstance(stored_muzzle_features, list):
                    print(f"   Muzzle features count: {len(stored_muzzle_features)}")
                    if len(stored_muzzle_features) > 0:
                        first_feature = stored_muzzle_features[0]
                        print(f"   First feature type: {type(first_feature)}")
                        if isinstance(first_feature, dict):
                            print(f"   First feature keys: {list(first_feature.keys())}")
                            if 'muzzle_features' in first_feature:
                                print(f"   ✅ Has nested muzzle_features")
                            else:
                                print(f"   ❌ No nested muzzle_features")
                else:
                    print(f"   Muzzle features: {stored_muzzle_features}")
                
                # Check if this pet has the right animal type for comparison
                query_animal_types = [animal['animal_type'] for animal in query_animal_data]
                stored_animal_type = stored_pet.get('animal_type', 'unknown')
                print(f"   Query animal types: {query_animal_types}")
                print(f"   Stored animal type: {stored_animal_type}")
                print(f"   Type match: {'✅ YES' if stored_animal_type in query_animal_types else '❌ NO'}")
                
                # Handle the data structure for pets stored via Python backend
                # Check if this pet has muzzle_data (Python backend) or muzzle_features (Flutter direct)
                if 'muzzle_data' in stored_pet:
                    # Pet was stored via Python backend - use muzzle_data
                    stored_muzzle_features = stored_pet.get('muzzle_data', [])
                    print(f"   Using muzzle_data from Python backend")
                else:
                    # Pet was stored directly via Flutter - use muzzle_features
                    stored_muzzle_features = stored_pet.get('muzzle_features', [])
                    print(f"   Using muzzle_features from Flutter")
                
                # If it's a single animal (not a list), wrap it in a list
                if not isinstance(stored_muzzle_features, list):
                    stored_muzzle_features = [stored_muzzle_features]
                
                # Compare each detected animal with stored animals
                print(f"\n   🔍 COMPARISON ANALYSIS:")
                print(f"   " + "-" * 30)
                
                for query_animal in query_animal_data:
                    print(f"\n   Query Animal: {query_animal['animal_type']} (confidence: {query_animal['confidence']:.3f})")
                    
                    # Check if animal types are compatible for comparison
                    query_type = query_animal['animal_type'].lower()
                    stored_type = stored_pet.get('animal_type', 'unknown').lower()
                    
                    # Flexible animal type matching - allow similar animals to be compared
                    compatible_types = {
                        'cat': ['cat', 'dog', 'pet', 'animal'],  # Cats can match with dogs for general pet recognition
                        'dog': ['dog', 'cat', 'pet', 'animal'],  # Dogs can match with cats for general pet recognition
                        'horse': ['horse', 'pet', 'animal'],
                        'cow': ['cow', 'pet', 'animal'],
                        'sheep': ['sheep', 'pet', 'animal']
                    }
                    
                    # Check if types are compatible
                    animal_type_match = (
                        query_type == stored_type or  # Exact match
                        stored_type in compatible_types.get(query_type, []) or  # Compatible types
                        query_type in compatible_types.get(stored_type, []) or  # Reverse compatibility
                        stored_type in ['pet', 'animal', 'unknown']  # Generic types
                    )
                    
                    has_muzzle_features = 'muzzle_features' in query_animal
                    
                    print(f"   Query type: {query_type}, Stored type: {stored_type}")
                    print(f"   Animal type compatible: {'✅ YES' if animal_type_match else '❌ NO'}")
                    print(f"   Has muzzle features: {'✅ YES' if has_muzzle_features else '❌ NO'}")
                    
                    if animal_type_match and has_muzzle_features:
                        print(f"   ✅ Proceeding with feature comparison...")
                        
                        # Compare with stored animals - now both are in the same flattened format
                        for stored_animal in stored_muzzle_features:
                            if stored_animal:
                                print(f"   🔍 Comparing features for: {stored_pet.get('name', 'Unknown')}")
                                
                                # Log feature structure comparison
                                query_features = query_animal['muzzle_features'] if 'muzzle_features' in query_animal else query_animal
                                
                                # Extract stored features - handle nested structure from Flutter
                                if isinstance(stored_animal, dict):
                                    if 'muzzle_features' in stored_animal:
                                        stored_features = stored_animal['muzzle_features']
                                        print(f"   ✅ Extracted nested muzzle_features from stored_animal")
                                    else:
                                        # If no nested key, assume stored_animal IS the features
                                        stored_features = stored_animal
                                        print(f"   ⚠️ No nested muzzle_features key, using stored_animal directly")
                                else:
                                    stored_features = stored_animal
                                    print(f"   ⚠️ stored_animal is not a dict, using as-is")
                                
                                print(f"   Query features keys: {list(query_features.keys()) if isinstance(query_features, dict) else 'Not a dict'}")
                                print(f"   Stored features keys: {list(stored_features.keys()) if isinstance(stored_features, dict) else 'Not a dict'}")
                                
                                # Debug: Check if features have the expected structure
                                if isinstance(stored_features, dict):
                                    has_traditional = 'traditional_features' in stored_features
                                    has_sift = 'sift_features' in stored_features
                                    has_dogfacenet = 'dogfacenet_embeddings' in stored_features
                                    print(f"   Feature availability - Traditional: {has_traditional}, SIFT: {has_sift}, DogFaceNet: {has_dogfacenet}")
                                
                                # Debug: Check DogFaceNet preservation before comparison
                                debug_dogfacenet_preservation([{'muzzle_features': stored_features}], "stored_from_db")
                                debug_dogfacenet_preservation([{'muzzle_features': query_features}], "query_converted")
                                
                                print(f"   🚀 PROCEEDING WITH DOGFACENET-ENHANCED COMPARISON!")
                                print(f"   " + "=" * 50)
                                
                                # Use the DogFaceNet-enhanced feature comparison
                                similarity, is_match = calculate_flattened_similarity_with_dogfacenet(
                                    query_features,
                                    stored_features
                                )

                                print(f"   Similarity score: {similarity:.4f}, Match: {is_match}")
                                
                                # Show feature usage summary
                                print(f"   📊 FEATURE USAGE SUMMARY:")
                                if 'dogfacenet_embeddings' in query_features and 'dogfacenet_embeddings' in stored_features:
                                    print(f"      🐕 DogFaceNet: ✅ Used (128-dim embeddings)")
                                else:
                                    print(f"      🐕 DogFaceNet: ❌ Not available")
                                
                                if 'sift_features' in query_features and 'sift_features' in stored_features:
                                    print(f"      🔍 SIFT: ✅ Used (keypoint matching)")
                                else:
                                    print(f"      🔍 SIFT: ❌ Not available")
                                
                                if 'traditional_features' in query_features and 'traditional_features' in stored_features:
                                    print(f"      📏 Traditional: ✅ Used (texture/LBP features)")
                                else:
                                    print(f"      📏 Traditional: ❌ Not available")

                                if is_match:
                                    print(f"   🎉 MATCH FOUND!")
                                    matches.append({
                                        'pet_id': stored_pet_doc.id,
                                        'pet_name': stored_pet.get('name', 'Unknown'),
                                        'breed': stored_pet.get('breed', 'Unknown'),
                                        'owner': stored_pet.get('owner', 'Unknown'),
                                        'owner_name': stored_pet.get('owner_name', 'Unknown'),
                                        'owner_email': stored_pet.get('owner_email', 'Unknown'),
                                        'owner_phone': stored_pet.get('owner_phone', 'Unknown'),
                                        'similarity_score': round(similarity, 4),
                                        'animal_type': query_animal['animal_type'],
                                        'confidence': query_animal['confidence']
                                    })
                                else:
                                    print(f"   ❌ No match (similarity too low)")
                            else:
                                print(f"   ❌ Stored animal is empty or None")
                    else:
                        print(f"   ❌ Skipping comparison (type mismatch or no features)")

        except Exception as e:
            print(f"❌ Database query error: {e}")
            import traceback
            traceback.print_exc()
            return jsonify({"message": "Error querying database"}), 500

        print(f"\n📊 FINAL RESULTS:")
        print("-" * 30)
        print(f"Total pets in database: {pet_count}")
        print(f"Matches found: {len(matches)}")
        
        if matches:
            print(f"\n🎯 MATCHES DETAILS:")
            print("-" * 25)
            # Debug: Show match details
            for i, match in enumerate(matches):
                print(f"Match {i+1}: {match['pet_name']} (similarity: {match['similarity_score']:.4f})")
                print(f"   Pet ID: {match['pet_id']}")
                print(f"   Breed: {match['breed']}")
                print(f"   Owner: {match['owner_name']}")
                print(f"   Animal Type: {match['animal_type']}")
        else:
            print(f"\n❌ NO MATCHES FOUND")
            print("Possible reasons:")
            print("   - No pets in database with matching animal type")
            print("   - Feature extraction failed")
            print("   - Data structure mismatch")
            print("   - Similarity threshold too high")

        # Sort by similarity
        matches.sort(key=lambda x: x['similarity_score'], reverse=True)

        print(f"\n🚀 Returning response...")
        return jsonify({
            "message": "Pet identification completed",
            "query_animals_detected": len(query_animal_data),
            "matches_found": len(matches),
            "matches": matches[:5]  # Return top 5 matches
        }), 200

    except Exception as e:
        print(f"Error in identify_pet: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"message": f"Error: {e}"}), 500


# Enhanced Feature Extractor with DogFaceNet Integration
class SuperEnhancedMuzzleFeatureExtractor(ImprovedMuzzleFeatureExtractor):
    """
    Enhanced version that combines your existing features with DogFaceNet
    """
    
    def __init__(self):
        super().__init__()
        self.dogfacenet = DogFaceNetEnhancer()
        
        print("🚀 Super Enhanced Feature Extractor initialized!")
        print(f"   - Traditional features: ✅")
        print(f"   - SIFT features: ✅")
        print(f"   - DogFaceNet: {'✅' if TENSORFLOW_AVAILABLE else '❌'}")
    
    def extract_super_enhanced_features(self, image, animal_box):
        """
        Extract features using multiple methods:
        1. Your existing comprehensive features
        2. DogFaceNet embeddings
        3. SIFT features
        4. Combined similarity scoring
        """
        try:
            print("🔍 Starting super enhanced feature extraction...")
            
            # Get your existing features (already excellent!)
            existing_features = self.extract_comprehensive_features(image, animal_box)
            
            # Extract DogFaceNet embeddings (if available)
            dogfacenet_embeddings = None
            if TENSORFLOW_AVAILABLE:
                dogfacenet_embeddings = self.dogfacenet.extract_dog_embeddings(image, animal_box)
            
            # Extract SIFT features (your existing implementation)
            sift_features = sift_enhancer.extract_sift_features_from_muzzle(image, animal_box)
            
            if existing_features and (dogfacenet_embeddings or sift_features):
                combined_features = {
                    'traditional_features': existing_features,
                    'dogfacenet_embeddings': dogfacenet_embeddings,
                    'sift_features': sift_features,
                    'feature_types': ['traditional', 'sift'] + (['dogfacenet'] if dogfacenet_embeddings else []),
                    'has_dogfacenet': dogfacenet_embeddings is not None,
                    'has_sift': sift_features is not None,
                    'embedding_size': len(dogfacenet_embeddings) if dogfacenet_embeddings else 0
                }
                
                print(f"✅ Super enhanced features extracted successfully!")
                print(f"   - Traditional features: ✓ ({existing_features.get('feature_dimension', 'N/A')} dimensions)")
                print(f"   - DogFaceNet embeddings: {'✓' if dogfacenet_embeddings else '✗'} (size: {len(dogfacenet_embeddings) if dogfacenet_embeddings else 0})")
                print(f"   - SIFT features: {'✓' if sift_features else '✗'} ({sift_features.get('keypoints_count', 0) if sift_features else 0} keypoints)")
                
                return combined_features
                
            elif existing_features:
                # Fallback to existing features
                print("⚠️ DogFaceNet/SIFT extraction failed, using traditional features")
                return {
                    'traditional_features': existing_features,
                    'dogfacenet_embeddings': None,
                    'sift_features': sift_features,
                    'feature_types': ['traditional'] + (['sift'] if sift_features else []),
                    'has_dogfacenet': False,
                    'has_sift': sift_features is not None
                }
            else:
                print("❌ All feature extraction methods failed")
                return None
                
        except Exception as e:
            print(f"❌ Error in super enhanced feature extraction: {e}")
            return None


def calculate_super_enhanced_similarity(features1, features2, threshold=0.80):
    """
    Calculate similarity using all available feature types with intelligent weighting
    """
    try:
        print(f"\n🔍 SUPER ENHANCED SIMILARITY CALCULATION:")
        print(f"   " + "-" * 50)
        
        if not features1 or not features2:
            print("   ❌ One or both feature sets are empty")
            return 0.0, False
        
        print(f"   ✅ Both feature sets present")
        
        similarities = {}
        weights = {}
        
        # 1. Traditional features similarity (your existing method)
        if (features1.get('traditional_features') and features2.get('traditional_features')):
            trad_sim, trad_match = calculate_improved_similarity(
                features1['traditional_features'], 
                features2['traditional_features'],
                threshold
            )
            similarities['traditional'] = trad_sim
            weights['traditional'] = 0.25
            print(f"   Traditional similarity: {trad_sim:.4f}")
        
        # 2. DogFaceNet similarity (new!)
        if (TENSORFLOW_AVAILABLE and 
            features1.get('dogfacenet_embeddings') and 
            features2.get('dogfacenet_embeddings')):
            
            # Create DogFaceNet enhancer instance for similarity calculation
            dogfacenet_enhancer = DogFaceNetEnhancer()
            dogfacenet_sim, dogfacenet_match = dogfacenet_enhancer.calculate_dogfacenet_similarity(
                features1['dogfacenet_embeddings'],
                features2['dogfacenet_embeddings']
            )
            similarities['dogfacenet'] = dogfacenet_sim
            weights['dogfacenet'] = 0.45  # Highest weight for specialized model
            print(f"   DogFaceNet similarity: {dogfacenet_sim:.4f}")
        
        # 3. SIFT similarity (your existing method)
        if (features1.get('sift_features') and features2.get('sift_features')):
            sift_sim, sift_match = sift_enhancer.calculate_sift_similarity(
                features1['sift_features'],
                features2['sift_features'],
                threshold
            )
            similarities['sift'] = sift_sim
            weights['sift'] = 0.30
            print(f"   SIFT similarity: {sift_sim:.4f}")
        
        # Calculate weighted combination
        if similarities:
            # Normalize weights
            total_weight = sum(weights[k] for k in similarities.keys())
            
            combined_similarity = sum(
                similarities[method] * (weights[method] / total_weight)
                for method in similarities.keys()
            )
            
            # Adjust threshold based on available methods
            adjusted_threshold = threshold
            if 'dogfacenet' in similarities:
                adjusted_threshold = threshold - 0.05  # Slightly lower threshold with DogFaceNet
            
            is_match = combined_similarity >= adjusted_threshold
            
            print(f"   🎯 Combined similarity: {combined_similarity:.4f}")
            print(f"   Methods used: {list(similarities.keys())}")
            print(f"   Weights: {[(k, f'{v/total_weight:.2f}') for k, v in weights.items() if k in similarities]}")
            print(f"   Match: {is_match} (threshold: {adjusted_threshold:.2f})")
            
            return combined_similarity, is_match
        else:
            print("   ❌ No valid similarity methods available")
            return 0.0, False
            
    except Exception as e:
        print(f"   ❌ Error in super enhanced similarity: {e}")
        return 0.0, False


# Integration with your existing system
def initialize_super_enhanced_system():
    """Initialize the super enhanced recognition system"""
    try:
        print("\n🚀 INITIALIZING SUPER ENHANCED DOG RECOGNITION SYSTEM")
        print("=" * 60)
        
        # Initialize enhanced feature extractor
        global super_feature_extractor
        
        print("1. Loading super enhanced feature extractor...")
        super_feature_extractor = SuperEnhancedMuzzleFeatureExtractor()
        
        print("✅ Super enhanced system initialized successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error initializing super enhanced system: {e}")
        return False


# Modified functions to use super enhanced features
def extract_super_enhanced_muzzle_features(image, animal_box):
    """Extract features using the super enhanced method"""
    try:
        return super_feature_extractor.extract_super_enhanced_features(image, animal_box)
    except Exception as e:
        print(f"⚠️ Super enhanced extraction failed, falling back to enhanced method: {e}")
        # Fallback to your existing method
        return extract_enhanced_muzzle_features(image, animal_box)


# Initialize the super enhanced system
print("\n🚀 Initializing machine learning enhancements...")
initialize_super_enhanced_system()


def debug_dogfacenet_preservation(animal_data, stage="unknown"):
    """Debug function to check if DogFaceNet embeddings are preserved"""
    print(f"\n🔍 DOGFACENET PRESERVATION CHECK - {stage.upper()}")
    print("=" * 50)
    
    for i, animal in enumerate(animal_data):
        print(f"\n🐾 Animal {i+1}: {animal.get('animal_type', 'unknown')}")
        
        if 'muzzle_features' in animal:
            mf = animal['muzzle_features']
            print(f"   Muzzle features keys: {list(mf.keys())}")
            
            # Check DogFaceNet embeddings
            if 'dogfacenet_embeddings' in mf:
                embeddings = mf['dogfacenet_embeddings']
                if embeddings:
                    print(f"   ✅ DogFaceNet embeddings: {len(embeddings)} dimensions")
                    print(f"   First 5 values: {embeddings[:5]}")
                    print(f"   Last 5 values: {embeddings[-5:]}")
                    print(f"   Value range: [{min(embeddings):.4f}, {max(embeddings):.4f}]")
                else:
                    print(f"   ❌ DogFaceNet embeddings: None/Empty")
            else:
                print(f"   ❌ No DogFaceNet embeddings key found")
            
            # Check metadata
            has_dogfacenet = mf.get('has_dogfacenet', False)
            print(f"   Has DogFaceNet flag: {has_dogfacenet}")
            
            if 'feature_types' in mf:
                feature_types = mf['feature_types']
                print(f"   Feature types: {feature_types}")
                print(f"   DogFaceNet in types: {'dogfacenet' in feature_types}")
        else:
            print(f"   ❌ No muzzle features found")


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
            # In development, you might want to skip verification
            # In production, ALWAYS verify webhook signatures
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
                # Update user subscription in Firestore
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
            'accountType': 'pro' if status == 'active' else 'basic',  # Update account type
            'subscription': {
                'is_pro': status == 'active',
                'status': status,
                'payment_intent_id': payment_intent_id,
                'expires_at': expires_at.isoformat(),
                'updated_at': datetime.now().isoformat(),
            },
            'kennel_capacity': 5 if status == 'active' else 3,  # PRO gets 5 pets
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
                # Update Firestore
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
# STRIPE CHECKOUT SESSION ROUTES (using stripe.py module)
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
        # If JSON request (from Flutter), return JSON. If form (from web), redirect
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
            # If session_id provided, get customer from session
            if session_id and not customer_id:
                checkout_session = stripe.checkout.Session.retrieve(session_id)
                customer_id = checkout_session.customer
        
        if not customer_id:
            return jsonify({"error": "customer_id is required"}), 400
        
        # Get return URL from request or use default
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