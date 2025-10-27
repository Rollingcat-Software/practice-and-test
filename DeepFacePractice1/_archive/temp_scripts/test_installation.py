"""
Test Installation Script
Run this to verify everything is set up correctly.
"""

import sys
from pathlib import Path

# Use ASCII-safe characters for Windows
OK = "[OK]"
FAIL = "[FAIL]"
WARN = "[WARN]"

print("=" * 60)
print("TESTING DEEPFACE PRACTICE PROJECT INSTALLATION")
print("=" * 60 + "\n")

# Test 1: Check Python version
print("1. Checking Python version...")
print(f"   Python {sys.version}")
if sys.version_info >= (3, 8):
    print(f"   {OK} Python version OK\n")
else:
    print(f"   {FAIL} Need Python 3.8 or higher\n")
    sys.exit(1)

# Test 2: Check required packages
print("2. Checking required packages...")
required_packages = [
    ('deepface', 'DeepFace'),
    ('tensorflow', 'TensorFlow'),
    ('cv2', 'OpenCV'),
    ('numpy', 'NumPy'),
    ('matplotlib', 'Matplotlib'),
]

all_installed = True
for module_name, display_name in required_packages:
    try:
        __import__(module_name)
        print(f"   {OK} {display_name}")
    except ImportError:
        print(f"   {FAIL} {display_name} NOT FOUND")
        all_installed = False

if not all_installed:
    print("\n   Please run: pip install -r requirements.txt\n")
    sys.exit(1)
else:
    print(f"   {OK} All packages installed\n")

# Test 3: Check project structure
print("3. Checking project structure...")
required_dirs = ['src', 'src/models', 'src/services', 'src/utils', 'src/demos', 'images', 'output']
required_files = [
    'src/__init__.py',
    'src/config.py',
    'src/models/verification_result.py',
    'src/services/face_verification_service.py',
    'src/utils/logger.py',
    'src/demos/demo_1_verification.py',
]

structure_ok = True
for dir_path in required_dirs:
    if Path(dir_path).exists():
        print(f"   {OK} {dir_path}/")
    else:
        print(f"   {FAIL} {dir_path}/ NOT FOUND")
        structure_ok = False

for file_path in required_files:
    if Path(file_path).exists():
        print(f"   {OK} {file_path}")
    else:
        print(f"   {FAIL} {file_path} NOT FOUND")
        structure_ok = False

if not structure_ok:
    print("\n   Project structure incomplete!\n")
    sys.exit(1)
else:
    print(f"   {OK} Project structure OK\n")

# Test 4: Check sample images
print("4. Checking sample images...")
sample_images = ['images/kisi_A_1.jpg', 'images/kisi_A_2.jpg', 'images/kisi_B_1.jpg']
images_ok = True
for img in sample_images:
    if Path(img).exists():
        print(f"   {OK} {img}")
    else:
        print(f"   {WARN} {img} NOT FOUND (optional)")
        images_ok = False

if images_ok:
    print(f"   {OK} All sample images present\n")
else:
    print(f"   {WARN} Some images missing (add them to run demos)\n")

# Test 5: Test imports
print("5. Testing module imports...")
try:
    from src.services.face_verification_service import FaceVerificationService
    from src.services.face_analysis_service import FaceAnalysisService
    from src.services.face_recognition_service import FaceRecognitionService
    from src.utils.visualizer import FaceVisualizer
    from src.utils.logger import setup_logger

    print(f"   {OK} All modules import successfully\n")
except Exception as e:
    print(f"   {FAIL} Import error: {e}\n")
    sys.exit(1)

# Test 6: Quick functionality test (if images available)
if images_ok:
    print("6. Running quick functionality test...")
    try:
        from src.services.face_verification_service import FaceVerificationService

        print("   Testing face verification...")
        service = FaceVerificationService(model_name="Facenet512", detector_backend="opencv")

        result = service.verify(
            "images/kisi_A_1.jpg",
            "images/kisi_A_2.jpg"
        )

        print(f"   {OK} Verification test passed!")
        print(f"      Same person detected: {result.verified}")
        print(f"      Distance: {result.distance:.4f}")
        print()

    except Exception as e:
        print(f"   {WARN} Functionality test failed: {e}")
        print(f"      (This might be normal on first run - DeepFace downloads models)\n")
else:
    print("6. Skipping functionality test (no sample images)\n")

# Summary
print("=" * 60)
print("INSTALLATION TEST COMPLETE")
print("=" * 60)
print(f"\n{OK} Your DeepFace Practice Project is ready to use!")
print("\nNext steps:")
print("  1. Read README.md for overview")
print("  2. Run: python quick_start.py")
print("  3. Or run individual demos:")
print("     - python src/demos/demo_1_verification.py")
print("     - python src/demos/demo_2_analysis.py")
print("     - python src/demos/demo_3_embeddings.py")
print("\nHappy learning!\n")
