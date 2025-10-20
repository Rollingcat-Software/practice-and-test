"""
Demo 4: Dynamic Multi-Person Recognition
Demonstrates the scalable person management system.

Key Learning Points:
- Dynamic person discovery (works with any number of people)
- Person database management
- Multi-person verification
- Scalable face recognition system
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.services import PersonManager, FaceVerificationService


def print_header(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(f"{title:^70}")
    print("=" * 70 + "\n")


def demo_4a_person_discovery():
    """
    Demo 4A: Automatic Person Discovery
    Shows how the system automatically finds all persons.
    """
    print_header("DEMO 4A: Automatic Person Discovery")

    manager = PersonManager()

    print("[*] Scanning images directory for persons...")
    persons = manager.scan_all_persons()

    print(f"\n[OK] Found {len(persons)} person(s) in the database")
    print("\n" + "-" * 70)
    print(f"{'Person ID':<15} {'Folder Name':<20} {'Images':<10}")
    print("-" * 70)

    for person in persons:
        print(f"{person.person_id:<15} {person.folder_name:<20} {person.image_count:<10}")

    print("-" * 70)

    # Show details for each person
    print("\nDetailed Information:")
    for person in persons:
        print(f"\n  [{person.folder_name}]")
        print(f"  Person ID: {person.person_id}")
        print(f"  Images: {person.image_count}")
        print("  Image Paths:")
        for img_path in person.image_paths:
            print(f"    - {Path(img_path).name}")

    return persons


def demo_4b_cross_person_verification(persons):
    """
    Demo 4B: Cross-Person Verification
    Verify that different persons are correctly identified as different.
    """
    print_header("DEMO 4B: Cross-Person Verification Matrix")

    if len(persons) < 2:
        print("[!] Need at least 2 persons for cross-verification")
        return

    print("This matrix shows verification results between all persons")
    print("Diagonal = same person (should be TRUE)")
    print("Off-diagonal = different persons (should be FALSE)\n")

    verifier = FaceVerificationService()

    # Create verification matrix
    n = len(persons)
    print(f"\n{'':<15}", end="")
    for p in persons:
        print(f"{p.folder_name:<15}", end="")
    print()
    print("-" * (15 + 15 * n))

    for i, person1 in enumerate(persons):
        print(f"{person1.folder_name:<15}", end="")

        for j, person2 in enumerate(persons):
            # Get first image of each person
            if not person1.image_paths or not person2.image_paths:
                print(f"{'NO IMG':<15}", end="")
                continue

            img1 = person1.image_paths[0]
            img2 = person2.image_paths[0]

            try:
                result = verifier.verify(img1, img2, enforce_detection=False)
                status = "MATCH" if result.verified else "DIFF"
                color = "[OK]" if (i == j) == result.verified else "[!!]"
                print(f"{color} {status:<10}", end="")
            except Exception as e:
                print(f"{'ERROR':<15}", end="")

        print()

    print("\nKey: [OK] = Expected result, [!!] = Unexpected result\n")


def demo_4c_intra_person_verification(persons):
    """
    Demo 4C: Intra-Person Verification
    Verify that multiple images of the same person match.
    """
    print_header("DEMO 4C: Intra-Person Verification")

    print("Testing if multiple images of the same person match each other\n")

    verifier = FaceVerificationService()

    for person in persons:
        if person.image_count < 2:
            print(f"[!] {person.folder_name}: Only 1 image (skipping)")
            continue

        print(f"\n[{person.folder_name}] - Testing {person.image_count} images")
        print("-" * 60)

        # Compare all pairs of images for this person
        for i in range(len(person.image_paths)):
            for j in range(i + 1, len(person.image_paths)):
                img1_path = person.image_paths[i]
                img2_path = person.image_paths[j]

                img1_name = Path(img1_path).name
                img2_name = Path(img2_path).name

                try:
                    result = verifier.verify(img1_path, img2_path, enforce_detection=False)

                    status_icon = "[OK]" if result.verified else "[!!]"
                    status_text = "MATCH" if result.verified else "NO MATCH"

                    print(f"  {status_icon} {img1_name} vs {img2_name}: "
                          f"{status_text} (distance: {result.distance:.4f})")

                except Exception as e:
                    print(f"  [X] {img1_name} vs {img2_name}: ERROR - {e}")


def demo_4d_person_lookup(persons):
    """
    Demo 4D: Person Identification
    Given an image, find which person it belongs to.
    """
    print_header("DEMO 4D: Person Identification")

    if not persons:
        print("[!] No persons in database")
        return

    print("Scenario: Given a query image, identify which person it belongs to\n")

    # Use first person's first image as query
    query_person = persons[0]
    if not query_person.image_paths:
        print("[!] No images available for testing")
        return

    query_image = query_person.image_paths[0]
    print(f"Query Image: {Path(query_image).name}")
    print(f"(This image belongs to: {query_person.folder_name})\n")

    verifier = FaceVerificationService()

    print("Comparing against all persons in database:")
    print("-" * 70)
    print(f"{'Person':<20} {'Status':<15} {'Distance':<15} {'Confidence'}")
    print("-" * 70)

    results = []
    for person in persons:
        if not person.image_paths:
            continue

        # Compare against first image of each person
        reference_image = person.image_paths[0]

        try:
            result = verifier.verify(query_image, reference_image, enforce_detection=False)
            results.append((person, result))

            status = "MATCH" if result.verified else "NO MATCH"
            print(f"{person.folder_name:<20} {status:<15} "
                  f"{result.distance:<15.4f} {result.confidence_percentage:.2f}%")

        except Exception as e:
            print(f"{person.folder_name:<20} ERROR")

    print("-" * 70)

    # Find best match
    if results:
        best_match = min(results, key=lambda x: x[1].distance)
        print(f"\n[OK] Best Match: {best_match[0].folder_name} "
              f"(distance: {best_match[1].distance:.4f})")

        if best_match[0].person_id == query_person.person_id:
            print("[OK] Correctly identified the person!")
        else:
            print("[!!] Incorrect identification!")


def demo_4e_scalability_test():
    """
    Demo 4E: Scalability Demonstration
    Show how easy it is to add new persons.
    """
    print_header("DEMO 4E: Scalability & Growth")

    manager = PersonManager()
    current_count = manager.get_person_count()

    print(f"Current Database: {current_count} persons")
    print("\nHow to add more persons:")
    print("-" * 70)

    print("\n1. Simple method - Add folders manually:")
    print("   images/")
    print("   ├── person_0001/  (existing)")
    print("   ├── person_0002/  (existing)")
    print("   ├── person_0003/  <- Add this folder")
    print("   │   ├── img_001.jpg  <- Add images here")
    print("   │   └── img_002.jpg")
    print("   ├── person_0004/  <- Add another")
    print("   │   └── img_001.jpg")
    print("   ...")
    print("   └── person_9999/  <- Can scale to 9,999 people!")

    print("\n2. Programmatic method - Use PersonManager:")
    print("   >>> manager = PersonManager()")
    print("   >>> manager.create_person(person_id=3, name='John Doe')")
    print("   >>> manager.add_image_to_person(3, 'path/to/photo.jpg')")

    print("\n3. Naming convention ensures proper sorting:")
    print("   person_0001, person_0002, ..., person_0999, person_1000")
    print("   Always sorts correctly! No person1, person10, person2 issues.")

    print("\n" + "=" * 70)
    print("System supports up to 9,999 persons with current configuration")
    print("Can be extended to 999,999 persons by changing PERSON_ID_DIGITS")
    print("=" * 70)


def main():
    """Run all dynamic person recognition demos."""
    print("\n" + "=" * 70)
    print(" " * 15 + "DYNAMIC MULTI-PERSON RECOGNITION")
    print(" " * 10 + "Scalable Face Recognition System Demo")
    print("=" * 70)

    try:
        # Demo 4A: Person Discovery
        persons = demo_4a_person_discovery()

        if not persons:
            print("\n[!] No persons found in database!")
            print("[*] Please add person folders to the 'images' directory")
            return

        # Demo 4B: Cross-Person Verification
        input("\nPress Enter to continue to Demo 4B (Cross-Person Verification)...")
        demo_4b_cross_person_verification(persons)

        # Demo 4C: Intra-Person Verification
        if any(p.image_count > 1 for p in persons):
            input("\nPress Enter to continue to Demo 4C (Intra-Person Verification)...")
            demo_4c_intra_person_verification(persons)

        # Demo 4D: Person Lookup
        input("\nPress Enter to continue to Demo 4D (Person Identification)...")
        demo_4d_person_lookup(persons)

        # Demo 4E: Scalability
        input("\nPress Enter to continue to Demo 4E (Scalability Info)...")
        demo_4e_scalability_test()

        print_header("TUTORIAL COMPLETE!")
        print("Key Takeaways:")
        print("  - System automatically discovers persons")
        print("  - Scales to thousands of people")
        print("  - Simple folder-based organization")
        print("  - Easy to add/remove persons")
        print("  - Professional naming convention")

    except KeyboardInterrupt:
        print("\n\n[!] Demo interrupted by user")
    except Exception as e:
        print(f"\n[X] Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
