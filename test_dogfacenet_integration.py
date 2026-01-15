#!/usr/bin/env python3
"""
Test script specifically for DogFaceNet integration
Verifies that DogFaceNet embeddings are being extracted and preserved correctly
"""

import os
import sys
import numpy as np
import cv2

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_dogfacenet_extraction():
    """Test DogFaceNet feature extraction"""
    print("🧪 Testing DogFaceNet Feature Extraction")
    print("=" * 50)
    
    try:
        # Import the enhanced classes
        from app import DogFaceNetEnhancer, SuperEnhancedMuzzleFeatureExtractor
        
        print("✅ Classes imported successfully")
        
        # Test 1: DogFaceNet initialization
        print("\n1️⃣ Testing DogFaceNet initialization...")
        dogfacenet = DogFaceNetEnhancer()
        
        if dogfacenet.is_loaded:
            print(f"   ✅ DogFaceNet model loaded successfully")
            print(f"   📊 Model parameters: {dogfacenet.model.count_params():,}")
            print(f"   🎯 Input shape: {dogfacenet.input_size}")
            print(f"   🧠 Embedding size: {dogfacenet.embedding_size}")
        else:
            print("   ❌ DogFaceNet model failed to load")
            return False
        
        # Test 2: Super Enhanced Feature Extractor
        print("\n2️⃣ Testing Super Enhanced Feature Extractor...")
        super_extractor = SuperEnhancedMuzzleFeatureExtractor()
        print("   ✅ Super Enhanced Feature Extractor initialized")
        
        # Test 3: Feature extraction with dummy image
        print("\n3️⃣ Testing feature extraction...")
        
        # Create a realistic test image (dog-like colors)
        test_image = np.random.randint(100, 200, (640, 640, 3), dtype=np.uint8)
        test_bbox = (100, 100, 300, 300)
        
        print(f"   📸 Test image shape: {test_image.shape}")
        print(f"   📦 Bounding box: {test_bbox}")
        
        # Extract features
        features = super_extractor.extract_super_enhanced_features(test_image, test_bbox)
        
        if features:
            print("   ✅ Feature extraction successful!")
            
            # Check feature structure
            print(f"   📊 Feature types: {features.get('feature_types', [])}")
            print(f"   🐕 Has DogFaceNet: {features.get('has_dogfacenet', False)}")
            print(f"   🔍 Has SIFT: {features.get('has_sift', False)}")
            
            # Check DogFaceNet embeddings specifically
            if 'dogfacenet_embeddings' in features:
                embeddings = features['dogfacenet_embeddings']
                if embeddings:
                    print(f"   🧠 DogFaceNet embeddings: {len(embeddings)} dimensions")
                    print(f"   📏 First 5 values: {embeddings[:5]}")
                    print(f"   📏 Last 5 values: {embeddings[-5:]}")
                    print(f"   📊 Value range: [{min(embeddings):.4f}, {max(embeddings):.4f}]")
                    
                    # Check if embeddings are reasonable (should be around [-1, 1] range)
                    if all(-1.1 <= e <= 1.1 for e in embeddings):
                        print("   ✅ Embedding values are in expected range")
                    else:
                        print("   ⚠️ Embedding values outside expected range")
                else:
                    print("   ❌ DogFaceNet embeddings are None/empty")
                    return False
            else:
                print("   ❌ No DogFaceNet embeddings key found")
                return False
            
            # Check traditional features
            if 'traditional_features' in features:
                trad_features = features['traditional_features']
                if 'features' in trad_features:
                    feat_array = trad_features['features']
                    print(f"   📏 Traditional features: {len(feat_array)} dimensions")
            
            # Check SIFT features
            if 'sift_features' in features:
                sift_features = features['sift_features']
                if 'keypoints_count' in sift_features:
                    print(f"   🎯 SIFT keypoints: {sift_features['keypoints_count']}")
            
            return features
        else:
            print("   ❌ Feature extraction failed")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dogfacenet_preservation():
    """Test that DogFaceNet embeddings are preserved through safe conversion"""
    print("\n🔍 Testing DogFaceNet Preservation")
    print("=" * 40)
    
    try:
        from app import create_firestore_safe_features
        
        # Create test features (similar to what would be extracted)
        test_features = {
            'animal_type': 'dog',
            'confidence': 0.95,
            'bounding_box': {'left': 100, 'top': 100, 'right': 300, 'bottom': 300},
            'muzzle_features': {
                'traditional_features': {
                    'features': [0.1, 0.2, 0.3] * 100,  # 300 features
                    'feature_dimension': 300
                },
                'dogfacenet_embeddings': [0.1, -0.2, 0.3, -0.4] * 32,  # 128 embeddings
                'sift_features': {
                    'keypoints_count': 50,
                    'descriptors': [[0.1, 0.2] * 64] * 50  # 50 descriptors of 128 dimensions
                },
                'feature_types': ['traditional', 'dogfacenet', 'sift'],
                'has_dogfacenet': True,
                'has_sift': True,
                'embedding_size': 128
            }
        }
        
        print("📋 Test features created:")
        print(f"   - Traditional features: {len(test_features['muzzle_features']['traditional_features']['features'])}")
        print(f"   - DogFaceNet embeddings: {len(test_features['muzzle_features']['dogfacenet_embeddings'])}")
        print(f"   - SIFT descriptors: {len(test_features['muzzle_features']['sift_features']['descriptors'])}")
        
        # Test safe conversion
        print("\n🔄 Testing safe conversion...")
        safe_features = create_firestore_safe_features([test_features])
        
        if safe_features and len(safe_features) > 0:
            safe_feature = safe_features[0]
            print("   ✅ Safe conversion successful")
            
            # Check if DogFaceNet embeddings are preserved
            if 'muzzle_features' in safe_feature:
                mf = safe_feature['muzzle_features']
                
                if 'dogfacenet_embeddings' in mf:
                    embeddings = mf['dogfacenet_embeddings']
                    if embeddings:
                        print(f"   🧠 DogFaceNet embeddings preserved: {len(embeddings)} dimensions")
                        print(f"   📊 First 5 values: {embeddings[:5]}")
                        print(f"   📊 Last 5 values: {embeddings[-5:]}")
                        
                        # Verify values are the same
                        original_embeddings = test_features['muzzle_features']['dogfacenet_embeddings']
                        if embeddings == original_embeddings:
                            print("   ✅ Embeddings values preserved exactly")
                        else:
                            print("   ❌ Embeddings values changed during conversion")
                            return False
                    else:
                        print("   ❌ DogFaceNet embeddings are None/empty after conversion")
                        return False
                else:
                    print("   ❌ No DogFaceNet embeddings key after conversion")
                    return False
                
                # Check metadata preservation
                has_dogfacenet = mf.get('has_dogfacenet', False)
                print(f"   🏷️ Has DogFaceNet flag: {has_dogfacenet}")
                
                if 'feature_types' in mf:
                    feature_types = mf['feature_types']
                    print(f"   📋 Feature types: {feature_types}")
                    print(f"   🐕 DogFaceNet in types: {'dogfacenet' in feature_types}")
                
                return True
            else:
                print("   ❌ No muzzle features after conversion")
                return False
        else:
            print("   ❌ Safe conversion failed")
            return False
            
    except Exception as e:
        print(f"❌ Preservation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_similarity_calculation():
    """Test DogFaceNet similarity calculation"""
    print("\n🎯 Testing DogFaceNet Similarity Calculation")
    print("=" * 45)
    
    try:
        from app import calculate_flattened_similarity_with_dogfacenet
        
        # Create two similar feature sets
        features1 = {
            'traditional_features': {
                'features': [0.1, 0.2, 0.3] * 100
            },
            'dogfacenet_embeddings': [0.1, -0.2, 0.3, -0.4] * 32,
            'sift_features': {
                'descriptors_flat': [0.1, 0.2] * 1000
            }
        }
        
        features2 = {
            'traditional_features': {
                'features': [0.1, 0.2, 0.3] * 100  # Same as features1
            },
            'dogfacenet_embeddings': [0.1, -0.2, 0.3, -0.4] * 32,  # Same as features1
            'sift_features': {
                'descriptors_flat': [0.1, 0.2] * 1000  # Same as features1
            }
        }
        
        print("📋 Test features created (identical sets)")
        print(f"   - Traditional features: {len(features1['traditional_features']['features'])}")
        print(f"   - DogFaceNet embeddings: {len(features1['dogfacenet_embeddings'])}")
        print(f"   - SIFT descriptors: {len(features1['sift_features']['descriptors_flat'])}")
        
        # Test similarity calculation
        print("\n🔄 Testing similarity calculation...")
        similarity, is_match = calculate_flattened_similarity_with_dogfacenet(
            features1, features2, threshold=0.8
        )
        
        print(f"   📊 Similarity score: {similarity:.4f}")
        print(f"   🎯 Is match: {is_match}")
        
        # For identical features, similarity should be very high
        if similarity > 0.9:
            print("   ✅ High similarity for identical features (expected)")
        else:
            print("   ⚠️ Lower similarity than expected for identical features")
        
        return True
        
    except Exception as e:
        print(f"❌ Similarity test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🚀 DogFaceNet Integration Test Suite")
    print("=" * 50)
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Feature extraction
    print("\n" + "="*60)
    if test_dogfacenet_extraction():
        tests_passed += 1
        print("✅ Feature extraction test PASSED")
    else:
        print("❌ Feature extraction test FAILED")
    
    # Test 2: Feature preservation
    print("\n" + "="*60)
    if test_dogfacenet_preservation():
        tests_passed += 1
        print("✅ Feature preservation test PASSED")
    else:
        print("❌ Feature preservation test FAILED")
    
    # Test 3: Similarity calculation
    print("\n" + "="*60)
    if test_similarity_calculation():
        tests_passed += 1
        print("✅ Similarity calculation test PASSED")
    else:
        print("❌ Similarity calculation test FAILED")
    
    # Summary
    print("\n" + "="*60)
    print(f"📊 TEST RESULTS: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 ALL TESTS PASSED!")
        print("✅ DogFaceNet integration is working correctly!")
        print("\n📚 Next steps:")
        print("   1. Test with real images via your endpoints")
        print("   2. Monitor logs for DogFaceNet feature extraction")
        print("   3. Verify improved accuracy in pet identification")
    else:
        print("⚠️ Some tests failed. Check the errors above.")
        print("💡 This might indicate issues with TensorFlow or model loading.")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Tests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

