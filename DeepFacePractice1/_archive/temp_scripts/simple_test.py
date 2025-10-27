"""
Simple Test - Just verify the structure is correct
"""

import sys
from pathlib import Path

print("=" * 60)
print("DEEPFACE PRACTICE PROJECT - SIMPLE TEST")
print("=" * 60 + "\n")

# Test imports
print("Testing imports...")
try:
    from src.services.face_verification_service import FaceVerificationService
    from src.services.face_analysis_service import FaceAnalysisService
    from src.services.face_recognition_service import FaceRecognitionService
    from src.models.verification_result import VerificationResult
    from src.models.face_analysis_result import FaceAnalysisResult
    from src.models.face_embedding import FaceEmbedding
    from src.utils.visualizer import FaceVisualizer
    from src.utils.logger import setup_logger
    from src.utils.file_helper import FileHelper
    from src.config import AppConfig

    print("[OK] All imports successful!\n")
except Exception as e:
    print(f"[FAIL] Import error: {e}\n")
    sys.exit(1)

# Test class instantiation
print("Testing service instantiation...")
try:
    verification_service = FaceVerificationService()
    print(f"  [OK] FaceVerificationService - Model: {verification_service.model_name}")

    analysis_service = FaceAnalysisService()
    print(f"  [OK] FaceAnalysisService - Detector: {analysis_service.detector_backend}")

    recognition_service = FaceRecognitionService()
    print(f"  [OK] FaceRecognitionService - Model: {recognition_service.model_name}")

    visualizer = FaceVisualizer()
    print(f"  [OK] FaceVisualizer - Figure size: {visualizer.figure_size}")

    logger = setup_logger()
    print(f"  [OK] Logger - Name: {logger.name}\n")

except Exception as e:
    print(f"[FAIL] Instantiation error: {e}\n")
    sys.exit(1)

# Test config
print("Testing configuration...")
try:
    config_default = AppConfig.default()
    print(f"  [OK] Default config - Model: {config_default.model.default_model}")

    config_fast = AppConfig.fast_mode()
    print(f"  [OK] Fast mode config - Model: {config_fast.model.default_model}")

    config_accurate = AppConfig.accurate_mode()
    print(f"  [OK] Accurate mode config - Model: {config_accurate.model.default_model}\n")

except Exception as e:
    print(f"[FAIL] Config error: {e}\n")
    sys.exit(1)

# Check file structure
print("Checking file structure...")
image_files = list(Path("images").glob("*.jpg"))
if image_files:
    print(f"  [OK] Found {len(image_files)} image(s) in images/")
    for img in image_files:
        print(f"       - {img.name}")
else:
    print("  [WARN] No images found in images/ folder")

print()

# Summary
print("=" * 60)
print("TEST COMPLETE - PROJECT STRUCTURE IS READY!")
print("=" * 60)
print("\nYour project is properly set up and ready to use.")
print("\nTo run demos:")
print("  python src/demos/demo_1_verification.py")
print("  python src/demos/demo_2_analysis.py")
print("  python src/demos/demo_3_embeddings.py")
print("\nOr use the interactive menu:")
print("  python quick_start.py")
print("\nNote: First run will download AI models (~100-500MB)")
print("      This is normal and only happens once.\n")
