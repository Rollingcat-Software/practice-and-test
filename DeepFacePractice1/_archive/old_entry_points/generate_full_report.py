"""
Generate Complete Visualization Report
Creates comprehensive visualizations for ALL persons and ALL comparisons.

Generates:
1. Individual person galleries
2. Verification grid showing all comparisons
3. Distance matrix heatmap
4. Interactive HTML report
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.services import PersonManager, FaceVerificationService
from src.utils.advanced_visualizer import AdvancedVisualizer
from src.utils.optimized_viewer import OptimizedViewer


def main():
    """Generate complete visualization report."""
    print("\n" + "=" * 70)
    print(" " * 15 + "COMPREHENSIVE VISUALIZATION REPORT")
    print(" " * 20 + "Generating all visualizations...")
    print("=" * 70 + "\n")

    # Initialize services
    manager = PersonManager()
    verifier = FaceVerificationService()
    visualizer = AdvancedVisualizer()
    optimized_viewer = OptimizedViewer()

    # Scan persons
    print("[*] Scanning persons...")
    persons = manager.scan_all_persons()

    if not persons:
        print("[!] No persons found. Please add person folders to images/")
        return

    print(f"[OK] Found {len(persons)} persons with {sum(p.image_count for p in persons)} total images\n")

    # Step 1: Generate person galleries
    print("=" * 70)
    print("STEP 1: Creating Individual Person Galleries")
    print("=" * 70 + "\n")

    for person in persons:
        print(f"[*] Creating gallery for {person.folder_name}...")
        visualizer.create_person_gallery(person)

    # Step 2: Run verifications and collect results
    print("\n" + "=" * 70)
    print("STEP 2: Running All Verifications")
    print("=" * 70 + "\n")

    verification_results = []

    # Intra-person verifications (same person)
    print("[*] Testing same-person comparisons...")
    for person in persons:
        if person.image_count < 2:
            continue

        for i in range(len(person.image_paths)):
            for j in range(i + 1, len(person.image_paths)):
                try:
                    result = verifier.verify(
                        person.image_paths[i],
                        person.image_paths[j],
                        enforce_detection=False
                    )
                    verification_results.append((person, person, result))
                except Exception as e:
                    print(f"  [!] Error: {e}")

    # Inter-person verifications (different people)
    print("[*] Testing different-person comparisons...")
    for i, person1 in enumerate(persons):
        if not person1.image_paths:
            continue

        for person2 in persons[i + 1:]:
            if not person2.image_paths:
                continue

            try:
                result = verifier.verify(
                    person1.image_paths[0],
                    person2.image_paths[0],
                    enforce_detection=False
                )
                verification_results.append((person1, person2, result))
            except Exception as e:
                print(f"  [!] Error: {e}")

    print(f"[OK] Completed {len(verification_results)} verifications\n")

    # Step 3: Create verification grid
    print("=" * 70)
    print("STEP 3: Creating Verification Grid")
    print("=" * 70 + "\n")

    print("[*] Generating comparison grid...")
    visualizer.create_verification_grid(
        verification_results,
        persons,
        output_file="verification_grid_all.png"
    )

    # Step 4: Create heatmap
    print("\n" + "=" * 70)
    print("STEP 4: Creating Distance Matrix Heatmap")
    print("=" * 70 + "\n")

    print("[*] Generating heatmap...")
    visualizer.create_verification_matrix_heatmap(
        persons,
        verifier,
        output_file="verification_heatmap.png"
    )

    # Step 5: Create HTML report
    print("\n" + "=" * 70)
    print("STEP 5: Creating HTML Report")
    print("=" * 70 + "\n")

    print("[*] Generating HTML report...")
    visualizer.create_html_report(
        persons,
        verification_results,
        output_file="verification_report.html"
    )

    # Step 6: Create Optimized Interactive Viewer
    print("\n" + "=" * 70)
    print("STEP 6: Creating Optimized Interactive Viewer")
    print("=" * 70 + "\n")

    print("[*] Generating optimized viewer...")
    optimized_viewer.create_optimized_viewer(
        verification_results,
        output_file="viewer.html"
    )

    # Summary
    print("\n" + "=" * 70)
    print(" " * 25 + "REPORT COMPLETE!")
    print("=" * 70)

    print(f"\nGenerated Files:")
    print(f"  - {len(persons)} person galleries (gallery_person_XXXX.png)")
    print(f"  - 1 verification grid (verification_grid_all.png)")
    print(f"  - 1 distance heatmap (verification_heatmap.png)")
    print(f"  - 1 HTML report (verification_report.html)")
    print(f"  - 1 OPTIMIZED VIEWER (viewer.html) [RECOMMENDED!]")

    print(f"\nAll files saved to: output/")
    print(f"\n" + "=" * 70)
    print(f">>> START HERE: Open Optimized Viewer <<<")
    print(f"=" * 70)
    print(f"\n  FILE: output/viewer.html")
    print(f"\nFeatures:")
    print(f"  - Professional dark UI")
    print(f"  - Navigation always visible (bottom bar)")
    print(f"  - Images front and center")
    print(f"  - All info on one screen (no scrolling)")
    print(f"  - Keyboard shortcuts (Arrow keys, Home/End)")
    print(f"  - Clickable list sidebar")
    print(f"\nOther reports:")
    print(f"  - verification_report.html: Data table view")
    print(f"  - PNG files: Static charts/grids")

    # Print statistics
    print(f"\n" + "=" * 70)
    print("Statistics:")
    print(f"  Total Persons: {len(persons)}")
    print(f"  Total Images: {sum(p.image_count for p in persons)}")
    print(f"  Total Verifications: {len(verification_results)}")

    matches = sum(1 for _, _, r in verification_results if r.verified)
    print(f"  Matches: {matches}")
    print(f"  Different: {len(verification_results) - matches}")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[!] Report generation interrupted")
    except Exception as e:
        print(f"\n[X] Error: {e}")
        import traceback

        traceback.print_exc()
