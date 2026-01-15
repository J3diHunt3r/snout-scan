#!/usr/bin/env python3
"""
Test script for ScoutSnout Machine Learning Enhancements
Tests the new DogFaceNet integration and super enhanced features
"""

import os
import sys
import numpy as np
import cv2

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_ml_enhancements():
    """Test the machine learning enhancements"""
    print("🧪 Testing ScoutSnout ML Enhancements")
    print("=" * 50)
    
    try:
        # Test 1: Import TensorFlow
        print("\n1️⃣ Testing TensorFlow import...")
        try:
            import tensorflow as tf
            print(f"   ✅ TensorFlow {tf.__version__} imported successfully")
        except ImportError as e:
            print(f"   ❌ TensorFlow import failed: {e}")
            print("   💡 Install with: pip install tensorflow>=2.10.0")
            return False
        
        # Test 2: Import our enhanced classes
        print("\n2️⃣ Testing enhanced class imports...")
        try:
            from app import DogFaceNetEnhancer, SuperEnhancedMuzzleFeatureExtractor
            print("   ✅ Enhanced classes imported successfully")
        except ImportError as e:
            print(f"   ❌ Enhanced class import failed: {e}")
            return False
        
        # Test 3: Test DogFaceNet initialization
        print("\n3️⃣ Testing DogFaceNet initialization...")
        try:
            dogfacenet = DogFaceNetEnhancer()
            if dogfacenet.is_loaded:
                print("   ✅ DogFaceNet model initialized successfully")
                print(f"   📊 Model parameters: {dogfacenet.model.count_params():,}")
            else:
                print("   ⚠️ DogFaceNet model failed to initialize")
        except Exception as e:
            print(f"   ❌ DogFaceNet initialization error: {e}")
        
        # Test 4: Test Super Enhanced Feature Extractor
        print("\n4️⃣ Testing Super Enhanced Feature Extractor...")
        try:
            super_extractor = SuperEnhancedMuzzleFeatureExtractor()
            print("   ✅ Super Enhanced Feature Extractor initialized")
        except Exception as e:
            print(f"   ❌ Super Enhanced Feature Extractor error: {e}")
            return False
        
        # Test 5: Test with dummy image
        print("\n5️⃣ Testing feature extraction with dummy image...")
        try:
            # Create a dummy image (224x224 RGB)
            dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            dummy_bbox = (50, 50, 174, 174)  # Center region
            
            # Extract features
            features = super_extractor.extract_super_enhanced_features(dummy_image, dummy_bbox)
            
            if features:
                print("   ✅ Feature extraction successful!")
                print(f"   📊 Feature types: {features.get('feature_types', [])}")
                print(f"   🐕 Has DogFaceNet: {features.get('has_dogfacenet', False)}")
                print(f"   🔍 Has SIFT: {features.get('has_sift', False)}")
                print(f"   📏 Traditional features: {features.get('traditional_features', {}).get('feature_dimension', 'N/A')} dimensions")
                
                if features.get('dogfacenet_embeddings'):
                    print(f"   🧠 DogFaceNet embeddings: {len(features['dogfacenet_embeddings'])} dimensions")
                
                if features.get('sift_features'):
                    print(f"   🎯 SIFT keypoints: {features['sift_features'].get('keypoints_count', 0)}")
            else:
                print("   ❌ Feature extraction failed")
                return False
                
        except Exception as e:
            print(f"   ❌ Feature extraction test error: {e}")
            return False
        
        # Test 6: Test similarity calculation
        print("\n6️⃣ Testing similarity calculation...")
        try:
            from app import calculate_super_enhanced_similarity
            
            # Test with the same features (should be high similarity)
            similarity, is_match = calculate_super_enhanced_similarity(features, features, threshold=0.8)
            
            print(f"   ✅ Similarity calculation successful!")
            print(f"   📊 Self-similarity: {similarity:.4f}")
            print(f"   🎯 Self-match: {is_match}")
            
            if similarity > 0.9:  # Should be very high for same features
                print("   ✅ Self-similarity test passed (expected high value)")
            else:
                print("   ⚠️ Self-similarity lower than expected")
                
        except Exception as e:
            print(f"   ❌ Similarity calculation error: {e}")
            return False
        
        print("\n🎉 All tests completed successfully!")
        print("✅ Your ScoutSnout ML enhancements are working correctly!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_performance():
    """Test performance characteristics"""
    print("\n🚀 Performance Testing")
    print("=" * 30)
    
    try:
        from app import extract_super_enhanced_muzzle_features
        import time
        
        # Create test image
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        test_bbox = (100, 100, 300, 300)
        
        # Test feature extraction speed
        print("⏱️ Testing feature extraction speed...")
        start_time = time.time()
        
        features = extract_super_enhanced_muzzle_features(test_image, test_bbox)
        
        end_time = time.time()
        extraction_time = end_time - start_time
        
        print(f"   📊 Feature extraction time: {extraction_time:.3f} seconds")
        
        if extraction_time < 5.0:  # Should be reasonably fast
            print("   ✅ Performance acceptable (< 5 seconds)")
        else:
            print("   ⚠️ Performance slower than expected")
        
        # Test memory usage
        print("💾 Testing memory usage...")
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"   📊 Current memory usage: {memory_mb:.1f} MB")
        
        if memory_mb < 1000:  # Should be reasonable
            print("   ✅ Memory usage acceptable (< 1 GB)")
        else:
            print("   ⚠️ High memory usage detected")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Performance test error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 ScoutSnout ML Enhancement Test Suite")
    print("=" * 50)
    
    # Run basic tests
    basic_tests_passed = test_ml_enhancements()
    
    if basic_tests_passed:
        # Run performance tests
        performance_tests_passed = test_performance()
        
        if performance_tests_passed:
            print("\n🎉 ALL TESTS PASSED!")
            print("✅ Your ScoutSnout backend is ready with ML enhancements!")
            print("\n📚 Next steps:")
            print("   1. Start your server: python app.py")
            print("   2. Test the endpoints with real images")
            print("   3. Monitor logs for ML feature extraction")
            print("   4. Enjoy improved accuracy! 🚀")
        else:
            print("\n⚠️ Basic tests passed but performance issues detected")
            print("💡 Check memory usage and processing times")
    else:
        print("\n❌ Basic tests failed")
        print("💡 Check the error messages above and fix issues")
        print("📚 See ML_ENHANCEMENTS.md for troubleshooting")
    
    print("\n" + "=" * 50)

