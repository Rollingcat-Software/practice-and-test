"""
Quick Start Script
Run this to see a quick demonstration of all DeepFace capabilities.

This is your entry point to start learning!
"""

import sys
from pathlib import Path

# Ensure directories exist
Path("images").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
Path("database").mkdir(exist_ok=True)

print("\n" + " " * 30)
print(" " * 15 + "DEEPFACE PRACTICE PROJECT")
print(" " * 10 + "Learn Facial Recognition & Analysis with AI")
print(" " * 30 + "\n")

# Check for person folders (new structure) or old images
from src.services import PersonManager

manager = PersonManager()
persons = manager.scan_all_persons()

if not persons:
    print("[WARN]  SETUP REQUIRED")
    print("=" * 60)
    print("\nTo get started, you need to add person folders with images.")
    print("\nFolder structure:")
    print("  images/")
    print("  ├── person_0001/")
    print("  │   ├── img_001.jpg")
    print("  │   └── img_002.jpg")
    print("  └── person_0002/")
    print("      └── img_001.jpg")

    print("\nImage requirements:")
    print("  - Clear face photos")
    print("  - Each person folder should contain images of ONE person")
    print("  - Any face images work - use your own photos!")

    print("\n" + "=" * 60)
    print("\nOnce you've added person folders, run this script again:")
    print("  python quick_start.py")
    print("\n" + "=" * 60 + "\n")
    sys.exit(0)

# Images found - show menu
print(f"[OK] Found {len(persons)} person(s) with {sum(p.image_count for p in persons)} total images!")
print("\nWhat would you like to learn?\n")

print("Available Tutorials:")
print("-" * 60)
print("1. Face Verification - Compare two faces")
print("   Learn: Similarity metrics, models, detectors")
print("   Run: python src/demos/demo_1_verification.py")
print()
print("2. Face Analysis - Extract facial attributes")
print("   Learn: Age, gender, emotion, ethnicity detection")
print("   Run: python src/demos/demo_2_analysis.py")
print()
print("3. Face Embeddings & Recognition - The mathematics behind it")
print("   Learn: Vector representations, database search")
print("   Run: python src/demos/demo_3_embeddings.py")
print()
print("4. Dynamic Multi-Person Recognition - Scalable system [NEW!]")
print("   Learn: Person management, database operations, scaling")
print("   Run: python src/demos/demo_4_dynamic_recognition.py")
print()
print("5. Run ALL demos (comprehensive learning)")
print("-" * 60)

choice = input("\nEnter your choice (1-5) or 'q' to quit: ").strip()

if choice == 'q':
    print("\nGoodbye! Happy learning! \n")
    sys.exit(0)

elif choice == '1':
    print("\n>> Launching Demo 1: Face Verification...\n")
    import subprocess

    result = subprocess.run([sys.executable, "src/demos/demo_1_verification.py"])
    sys.exit(result.returncode)

elif choice == '2':
    print("\n>> Launching Demo 2: Face Analysis...\n")
    import subprocess

    result = subprocess.run([sys.executable, "src/demos/demo_2_analysis.py"])
    sys.exit(result.returncode)

elif choice == '3':
    print("\n>> Launching Demo 3: Face Embeddings & Recognition...\n")
    import subprocess

    result = subprocess.run([sys.executable, "src/demos/demo_3_embeddings.py"])
    sys.exit(result.returncode)

elif choice == '4':
    print("\n>> Launching Demo 4: Dynamic Multi-Person Recognition...\n")
    import subprocess

    result = subprocess.run([sys.executable, "src/demos/demo_4_dynamic_recognition.py"])
    sys.exit(result.returncode)

elif choice == '5':
    print("\n>> Launching ALL Demos...\n")
    print("This will take a few minutes. Grab a coffee! \n")
    import subprocess

    print("\n" + "=" * 60)
    print("DEMO 1: FACE VERIFICATION")
    print("=" * 60)
    subprocess.run([sys.executable, "src/demos/demo_1_verification.py"])

    print("\n" + "=" * 60)
    print("DEMO 2: FACE ANALYSIS")
    print("=" * 60)
    subprocess.run([sys.executable, "src/demos/demo_2_analysis.py"])

    print("\n" + "=" * 60)
    print("DEMO 3: FACE EMBEDDINGS & RECOGNITION")
    print("=" * 60)
    subprocess.run([sys.executable, "src/demos/demo_3_embeddings.py"])

    print("\n" + "=" * 60)
    print("DEMO 4: DYNAMIC MULTI-PERSON RECOGNITION")
    print("=" * 60)
    subprocess.run([sys.executable, "src/demos/demo_4_dynamic_recognition.py"])

else:
    print("\n[ERROR] Invalid choice. Please run again and select 1-5.\n")
    sys.exit(1)

print("\n" + "= " * 30)
print("\nCongratulations on completing the tutorial!")
print("\nNext Steps:")
print("  • Check the 'output/' folder for visualizations")
print("  • Read the code in 'src/' to understand implementation")
print("  • Experiment with different models and parameters")
print("  • Try your own face images")
print("  • Build your own face recognition application!")
print("\n📚 Read README.md for more details and examples")
print("\n= " * 30 + "\n")
