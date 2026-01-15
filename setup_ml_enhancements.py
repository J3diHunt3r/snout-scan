#!/usr/bin/env python3
"""
Setup script for ScoutSnout Machine Learning Enhancements
Automates the installation and configuration process
"""

import os
import sys
import subprocess
import platform

def print_header():
    """Print setup header"""
    print("🚀 ScoutSnout ML Enhancement Setup")
    print("=" * 50)
    print("This script will install and configure machine learning")
    print("enhancements for your ScoutSnout backend.")
    print()

def check_python_version():
    """Check Python version compatibility"""
    print("🐍 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"   ❌ Python {version.major}.{version.minor} detected")
        print("   💡 Python 3.8+ is required for TensorFlow")
        return False
    
    print(f"   ✅ Python {version.major}.{version.minor}.{version.micro} - Compatible")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    
    try:
        # Upgrade pip first
        print("   🔄 Upgrading pip...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                      check=True, capture_output=True)
        
        # Install TensorFlow
        print("   🧠 Installing TensorFlow...")
        subprocess.run([sys.executable, "-m", "pip", "install", "tensorflow>=2.10.0"], 
                      check=True, capture_output=True)
        
        # Install other ML dependencies
        print("   🔧 Installing other ML dependencies...")
        subprocess.run([sys.executable, "-m", "pip", "install", "keras>=2.10.0"], 
                      check=True, capture_output=True)
        
        # Install requirements.txt
        print("   📋 Installing from requirements.txt...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True)
        
        print("   ✅ All dependencies installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"   ❌ Dependency installation failed: {e}")
        print("   💡 Try installing manually: pip install tensorflow>=2.10.0")
        return False

def verify_installation():
    """Verify that everything is installed correctly"""
    print("\n🔍 Verifying installation...")
    
    try:
        # Test TensorFlow import
        print("   🧠 Testing TensorFlow...")
        import tensorflow as tf
        print(f"   ✅ TensorFlow {tf.__version__} imported successfully")
        
        # Test other imports
        print("   🔧 Testing other imports...")
        import numpy as np
        import cv2
        import torch
        print("   ✅ All core libraries imported successfully")
        
        # Test GPU availability
        if tf.config.list_physical_devices('GPU'):
            print("   🚀 GPU detected - TensorFlow will use GPU acceleration!")
        else:
            print("   💻 No GPU detected - TensorFlow will use CPU")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Import test failed: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    directories = ['models', 'uploads', 'snout_data']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"   ✅ Created directory: {directory}")
        else:
            print(f"   ℹ️ Directory exists: {directory}")

def run_tests():
    """Run the test suite"""
    print("\n🧪 Running tests...")
    
    try:
        result = subprocess.run([sys.executable, "test_ml_enhancements.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("   ✅ All tests passed!")
            return True
        else:
            print("   ❌ Some tests failed")
            print("   📋 Test output:")
            print(result.stdout)
            print("   ❌ Test errors:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"   ❌ Test execution failed: {e}")
        return False

def print_next_steps():
    """Print next steps for the user"""
    print("\n🎯 Setup Complete! Next Steps:")
    print("=" * 40)
    print("1. 🚀 Start your server:")
    print("   python app.py")
    print()
    print("2. 🔍 Test the endpoints:")
    print("   - POST /scanFace")
    print("   - POST /storeSnout")
    print("   - POST /identifyPet")
    print()
    print("3. 📊 Monitor the logs for:")
    print("   - DogFaceNet initialization")
    print("   - Super enhanced feature extraction")
    print("   - ML-enhanced similarity calculations")
    print()
    print("4. 📚 Read the documentation:")
    print("   ML_ENHANCEMENTS.md")
    print()
    print("5. 🧪 Run tests anytime:")
    print("   python test_ml_enhancements.py")
    print()
    print("🎉 Enjoy your enhanced ScoutSnout backend!")

def main():
    """Main setup function"""
    print_header()
    
    # Check Python version
    if not check_python_version():
        print("\n❌ Setup cannot continue. Please upgrade Python.")
        return False
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Dependency installation failed. Please check the errors above.")
        return False
    
    # Create directories
    create_directories()
    
    # Verify installation
    if not verify_installation():
        print("\n❌ Installation verification failed. Please check the errors above.")
        return False
    
    # Run tests
    if not run_tests():
        print("\n⚠️ Some tests failed, but setup completed. Check the test output above.")
        print("💡 You can still try running the server and see what happens.")
    
    # Print next steps
    print_next_steps()
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n✅ Setup completed successfully!")
        else:
            print("\n❌ Setup encountered issues. Please check the errors above.")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error during setup: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

