"""
Demo 1: Dynamic Face Verification
Compare ALL persons and ALL images automatically.

Educational Focus:
- Understanding verification vs recognition
- Different models and their trade-offs
- Distance metrics and thresholds
- Working with multiple persons dynamically
"""

import os
import sys
import time
from pathlib import Path

# Fix Windows encoding issues
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.services import FaceVerificationService, PersonManager
from src.utils.visualizer import FaceVisualizer
from src.utils.logger import setup_logger

# Setup
logger = setup_logger("Demo1_Verification")
visualizer = FaceVisualizer()


def demo_1a_intra_person_verification():
    """
    Demo 1A: Intra-Person Verification
    Compare multiple images of the SAME person.

    Learn: Images of the same person should MATCH (verified=True, low distance)
    """
    print("\n" + "=" * 70)
    print("DEMO 1A: Intra-Person Verification (Same Person)")
    print("=" * 70 + "\n")

    manager = PersonManager()
    persons = manager.scan_all_persons()

    if not persons:
        print("[!] No persons found. Please add person folders to images/")
        return

    service = FaceVerificationService()

    print("Testing: Do multiple images of the SAME person match?\n")
    print("-" * 70)

    for person in persons:
        if person.image_count < 2:
            print(f"[SKIP] {person.folder_name}: Only 1 image")
            continue

        print(f"\n[{person.folder_name}] Testing {person.image_count} images")
        print("-" * 70)

        # Compare all pairs of images for this person
        comparisons = 0
        matches = 0

        for i in range(len(person.image_paths)):
            for j in range(i + 1, len(person.image_paths)):
                img1 = person.image_paths[i]
                img2 = person.image_paths[j]

                img1_name = Path(img1).name
                img2_name = Path(img2).name

                try:
                    result = service.verify(img1, img2, enforce_detection=False)
                    comparisons += 1

                    if result.verified:
                        matches += 1
                        status = "[OK] MATCH"
                    else:
                        status = "[!!] NO MATCH"

                    print(f"  {status:15} {img1_name:12} vs {img2_name:12} "
                          f"| Distance: {result.distance:.4f}")

                    # Save first comparison visualization
                    if comparisons == 1:
                        save_path = f"output/verification_{person.folder_name}_same.png"
                        visualizer.visualize_verification(result, save_path, show=False)

                except Exception as e:
                    print(f"  [X] ERROR       {img1_name:12} vs {img2_name:12} | {e}")

        if comparisons > 0:
            success_rate = (matches / comparisons) * 100
            print(f"\nResult: {matches}/{comparisons} matched ({success_rate:.1f}%)")


def demo_1b_inter_person_verification():
    """
    Demo 1B: Inter-Person Verification
    Compare images from DIFFERENT persons.

    Learn: Images of different people should NOT match (verified=False, high distance)
    """
    print("\n" + "=" * 70)
    print("DEMO 1B: Inter-Person Verification (Different People)")
    print("=" * 70 + "\n")

    manager = PersonManager()
    persons = manager.scan_all_persons()

    if len(persons) < 2:
        print("[!] Need at least 2 persons. Please add more person folders.")
        return

    service = FaceVerificationService()

    print("Testing: Do images of DIFFERENT people NOT match?\n")
    print("-" * 70)

    comparisons = 0
    correct_rejections = 0

    # Compare first image of each person with first image of every other person
    for i, person1 in enumerate(persons):
        if not person1.image_paths:
            continue

        for person2 in persons[i + 1:]:
            if not person2.image_paths:
                continue

            img1 = person1.image_paths[0]
            img2 = person2.image_paths[0]

            img1_name = f"{person1.folder_name}/{Path(img1).name}"
            img2_name = f"{person2.folder_name}/{Path(img2).name}"

            try:
                result = service.verify(img1, img2, enforce_detection=False)
                comparisons += 1

                if not result.verified:
                    correct_rejections += 1
                    status = "[OK] DIFFERENT"
                else:
                    status = "[!!] FALSE MATCH"

                print(f"  {status:18} {img1_name:25} vs {img2_name:25} "
                      f"| Distance: {result.distance:.4f}")

                # Save first comparison visualization
                if comparisons == 1:
                    save_path = "output/verification_different_people.png"
                    visualizer.visualize_verification(result, save_path, show=False)

            except Exception as e:
                print(f"  [X] ERROR           {img1_name:25} vs {img2_name:25} | {e}")

    if comparisons > 0:
        accuracy = (correct_rejections / comparisons) * 100
        print(f"\n{'-' * 70}")
        print(f"Result: {correct_rejections}/{comparisons} correctly identified as different ({accuracy:.1f}%)")


def demo_1c_verification_matrix():
    """
    Demo 1C: Complete Verification Matrix
    Show verification results for ALL person pairs.

    Learn: Visual matrix showing which persons match (useful for debugging)
    """
    print("\n" + "=" * 70)
    print("DEMO 1C: Complete Verification Matrix")
    print("=" * 70 + "\n")

    manager = PersonManager()
    persons = manager.scan_all_persons()

    if not persons:
        print("[!] No persons found.")
        return

    service = FaceVerificationService()

    print("Matrix showing verification between all persons")
    print("Legend: [✓] = Match, [X] = Different, [-] = No image\n")

    # Print header
    print(f"{'':15}", end="")
    for person in persons:
        print(f"{person.folder_name:15}", end="")
    print()
    print("-" * (15 + 15 * len(persons)))

    # Print matrix
    for i, person1 in enumerate(persons):
        print(f"{person1.folder_name:15}", end="")

        for j, person2 in enumerate(persons):
            if not person1.image_paths or not person2.image_paths:
                print(f"{'[-]':15}", end="")
                continue

            img1 = person1.image_paths[0]
            img2 = person2.image_paths[0]

            try:
                result = service.verify(img1, img2, enforce_detection=False)

                # Diagonal = same person (should match)
                # Off-diagonal = different person (should not match)
                if i == j:
                    symbol = "[✓]" if result.verified else "[!!]"
                else:
                    symbol = "[X]" if not result.verified else "[!!]"

                dist_str = f"{result.distance:.3f}"
                print(f"{symbol} {dist_str:10}", end="")

            except Exception as e:
                print(f"{'[ERR]':15}", end="")

        print()

    print("\n[✓] = Correct match | [X] = Correct rejection | [!!] = Error")


def demo_1d_model_comparison():
    """
    Demo 1D: Compare Different Models
    Test the same image pair with different models.

    Learn: Different models have different accuracy/speed trade-offs
    """
    print("\n" + "=" * 70)
    print("DEMO 1D: Model Comparison")
    print("=" * 70 + "\n")

    manager = PersonManager()
    persons = manager.scan_all_persons()

    if not persons or not persons[0].image_paths:
        print("[!] No images to test")
        return

    # Get first two images from first person (or cross-person if only 1 image)
    person1 = persons[0]
    if person1.image_count >= 2:
        img1 = person1.image_paths[0]
        img2 = person1.image_paths[1]
        comparison_type = "Same person"
    elif len(persons) >= 2:
        img1 = persons[0].image_paths[0]
        img2 = persons[1].image_paths[0]
        comparison_type = "Different people"
    else:
        print("[!] Need at least 2 images or 2 persons")
        return

    print(f"Comparing: {Path(img1).parent.name}/{Path(img1).name} vs "
          f"{Path(img2).parent.name}/{Path(img2).name}")
    print(f"Type: {comparison_type}\n")

    models = ["OpenFace", "Facenet", "Facenet512", "ArcFace"]

    print(f"{'Model':15} {'Verified':12} {'Distance':12} {'Threshold':12} {'Time':10}")
    print("-" * 70)

    for model in models:
        try:
            service = FaceVerificationService(model_name=model)
            start = time.time()
            result = service.verify(img1, img2, enforce_detection=False)
            elapsed = time.time() - start

            verified_str = "MATCH" if result.verified else "NO MATCH"
            print(f"{model:15} {verified_str:12} {result.distance:12.4f} "
                  f"{result.threshold:12.4f} {elapsed:10.3f} s")

        except Exception as e:
            print(f"{model:15} ERROR: {str(e)[:40]}")

    print("\nKey Takeaways:")
    print("  - Faster models may be less accurate")
    print("  - Choose model based on your speed/accuracy needs")


def demo_1e_all_images_summary():
    """
    Demo 1E: Complete Summary
    Summary statistics for all images and all persons.
    """
    print("\n" + "=" * 70)
    print("DEMO 1E: Complete Database Summary")
    print("=" * 70 + "\n")

    manager = PersonManager()
    persons = manager.scan_all_persons()

    total_persons = len(persons)
    total_images = sum(p.image_count for p in persons)

    print(f"Database Statistics:")
    print(f"  Total Persons: {total_persons}")
    print(f"  Total Images: {total_images}")
    print(f"  Average Images per Person: {total_images / total_persons if total_persons > 0 else 0:.1f}")

    print(f"\nPer-Person Breakdown:")
    print(f"{'Person ID':15} {'Folder':20} {'Images':10}")
    print("-" * 50)

    for person in persons:
        print(f"{person.person_id:<15} {person.folder_name:20} {person.image_count:<10}")

    print("\nAll images are ready for verification!")
    print("Check the 'output/' folder for visualization results.")


def main():
    """Run all verification demos."""
    print("\n" + "=" * 70)
    print(" " * 20 + "FACE VERIFICATION TUTORIAL")
    print(" " * 15 + "Automatic verification for ALL persons")
    print("=" * 70)

    try:
        # Demo 1A: Same person comparisons
        demo_1a_intra_person_verification()

        # Demo 1B: Different person comparisons
        demo_1b_inter_person_verification()

        # Demo 1C: Verification matrix
        demo_1c_verification_matrix()

        # Demo 1D: Model comparison
        demo_1d_model_comparison()

        # Demo 1E: Summary
        demo_1e_all_images_summary()

        print("\n" + "=" * 70)
        print(" " * 25 + "TUTORIAL COMPLETE!")
        print(" " * 15 + "Check the 'output' folder for visualizations.")
        print("=" * 70 + "\n")

    except KeyboardInterrupt:
        print("\n\n[!] Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
