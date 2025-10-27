"""
Quality Inspection and Validation Tool
Professional tool for analyzing and validating photo quality across all persons.

This demonstrates:
- Photo model with comprehensive attributes
- Person model with relationship management
- ImageQualityValidator service
- ImageInspectionTool for visualization
- Professional reporting
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.models.photo import Photo
from src.models.person import Person
from src.services.person_manager import PersonManager
from src.services.image_quality_validator import ImageQualityValidator
from src.services.image_inspection_tool import ImageInspectionTool
from src.utils.logger import setup_logger

logger = setup_logger("QualityInspection")


def print_header(title: str) -> None:
    """Print formatted header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def print_section(title: str) -> None:
    """Print section separator."""
    print("\n" + "-" * 70)
    print(f"  {title}")
    print("-" * 70)


def main():
    """Run complete quality inspection workflow."""

    print("\n" * 2)
    print(" " * 15 + "=" * 60)
    print(" " * 15 + "  PHOTO QUALITY INSPECTION & VALIDATION SYSTEM")
    print(" " * 15 + "  Professional Face Recognition Quality Analysis")
    print(" " * 15 + "=" * 60)
    print("\n")

    # Initialize services
    person_manager = PersonManager()
    validator = ImageQualityValidator()
    inspector = ImageInspectionTool()

    # =======================================================================
    # STEP 1: Discover persons in database
    # =======================================================================
    print_header("STEP 1: Scanning Database")

    persons_data = person_manager.scan_all_persons()

    if not persons_data:
        print("[ERROR] No persons found in database!")
        print("\nPlease add images to 'images/person_XXXX/' folders.")
        return

    print(f"[OK] Found {len(persons_data)} person(s) in database\n")

    # =======================================================================
    # STEP 2: Convert to enhanced Person objects with Photo models
    # =======================================================================
    print_header("STEP 2: Loading Photos and Creating Professional Models")

    enhanced_persons = []

    for person_data in persons_data:
        print(f"\nProcessing {person_data.folder_name}...")

        # Create enhanced Person
        person = Person(
            person_id=person_data.person_id,
            folder_name=person_data.folder_name,
            name=f"Person {person_data.person_id}",
        )

        # Create Photo objects for each image
        for img_path in person_data.image_paths:
            try:
                photo = Photo.from_file_path(
                    file_path=Path(img_path),
                    person_id=person.person_id,
                    person_folder=person.folder_name
                )
                person.add_photo(photo)
                print(
                    f"  [+] Loaded: {photo.filename} ({photo.get_file_size_mb():.2f} MB, {photo.width}x{photo.height})")
            except Exception as e:
                logger.error(f"Failed to load {img_path}: {e}")
                print(f"  [!] Failed: {Path(img_path).name} - {e}")

        enhanced_persons.append(person)
        print(f"  [OK] Created Person object with {len(person.photos)} photos")

    # =======================================================================
    # STEP 3: Validate photo quality
    # =======================================================================
    print_header("STEP 3: Validating Photo Quality")

    print("Running quality validation on all photos...\n")

    total_photos = 0
    problematic_photos = 0

    for person in enhanced_persons:
        print(f"\n{person.folder_name}:")
        print(f"  Validating {len(person.photos)} photo(s)...")

        for photo in person.photos:
            try:
                # Validate quality
                metrics = validator.validate_photo(photo)

                # Update photo with metrics
                photo.mark_as_validated(metrics)

                total_photos += 1

                # Display result
                if metrics.is_acceptable():
                    status = f"[OK]  {metrics.quality_level.value.upper():<12}"
                else:
                    status = f"[!!]  {metrics.quality_level.value.upper():<12}"
                    problematic_photos += 1

                print(f"    {status} {photo.filename:<20} Score: {metrics.overall_score:5.1f}%")

                # Show issues if any
                if metrics.issues:
                    for issue in metrics.issues[:2]:  # Show first 2 issues
                        print(f"            └─ {issue}")

            except Exception as e:
                logger.error(f"Validation failed for {photo.photo_id}: {e}")
                photo.mark_as_failed(str(e))
                print(f"    [FAIL] {photo.filename} - {e}")

        # Refresh person statistics
        person.refresh_statistics()

    # =======================================================================
    # STEP 4: Generate Summary Report
    # =======================================================================
    print_header("STEP 4: Quality Summary Report")

    print(f"Total Photos: {total_photos}")
    print(
        f"Problematic Photos: {problematic_photos} ({problematic_photos / total_photos * 100:.1f}%)\n" if total_photos > 0 else "")

    print_section("Per-Person Statistics")

    for person in enhanced_persons:
        stats = person.statistics

        print(f"\n{person.folder_name} ({person.name})")
        print(f"  Total Photos: {stats.total_photos}")
        print(f"  Validated: {stats.validated_photos}")
        print(f"  Problematic: {stats.problematic_photos}")
        print(f"  Average Quality: {stats.average_quality_score:.1f}%")

        if stats.total_photos > 0:
            print(f"\n  Quality Distribution:")
            print(f"    Excellent: {stats.excellent_quality_count}")
            print(f"    Good:      {stats.good_quality_count}")
            print(f"    Fair:      {stats.fair_quality_count}")
            print(f"    Poor:      {stats.poor_quality_count}")
            print(f"    Very Poor: {stats.very_poor_quality_count}")

        # Get problematic photos
        problematic = person.get_problematic_photos()
        if problematic:
            print(f"\n  [WARN] Problematic Photos:")
            for photo in problematic[:3]:  # Show first 3
                print(f"    • {photo.filename}: {photo.get_quality_summary()}")

    # =======================================================================
    # STEP 5: Create Visual Inspections
    # =======================================================================
    print_header("STEP 5: Creating Visual Inspection Reports")

    output_dir = Path("output/inspection")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Generating inspection visualizations...\n")

    for person in enhanced_persons:
        # Skip if no photos with metrics
        if not any(p.quality_metrics for p in person.photos):
            continue

        try:
            print(f"  Creating inspection for {person.folder_name}...")
            report = inspector.inspect_person(person, show=False)

            print(f"    [OK] Report generated")
            print(f"        Total: {report.total_images}")
            print(f"        Problematic: {report.problematic_images}")

            if report.recommendations:
                print(f"        Recommendations:")
                for rec in report.recommendations[:2]:
                    print(f"          • {rec}")

        except Exception as e:
            logger.error(f"Failed to create inspection for {person.folder_name}: {e}")
            print(f"    [FAIL] {e}")

    # Create detailed inspection for first problematic photo found
    print("\n  Creating detailed photo inspections for problematic images...")

    inspected_count = 0
    for person in enhanced_persons:
        problematic = person.get_problematic_photos()
        for photo in problematic[:2]:  # Inspect first 2 problematic photos per person
            try:
                inspector.inspect_photo(photo, show=False)
                print(f"    [OK] Detailed inspection: {photo.photo_id}")
                inspected_count += 1
            except Exception as e:
                logger.error(f"Failed to inspect {photo.photo_id}: {e}")

    if inspected_count == 0:
        print("    [OK] No problematic photos to inspect (all photos are good quality!)")

    # =======================================================================
    # STEP 6: Recommendations
    # =======================================================================
    print_header("STEP 6: Recommendations")

    print("Based on quality analysis:\n")

    for person in enhanced_persons:
        validation = person.get_validation_status()
        if validation['recommended_actions']:
            print(f"{person.folder_name}:")
            for action in validation['recommended_actions']:
                print(f"  • {action}")
            print()

    # =======================================================================
    # FINAL SUMMARY
    # =======================================================================
    print_header("INSPECTION COMPLETE")

    print(f"[OK] Processed {len(enhanced_persons)} person(s)")
    print(f"[OK] Validated {total_photos} photo(s)")
    print(f"[OK] Found {problematic_photos} problematic photo(s)")
    print(f"\n[OK] Inspection reports saved to: {output_dir.absolute()}")

    print("\n" + "=" * 70)
    print("\nNEXT STEPS:")
    print("  1. Review inspection visualizations in 'output/inspection/' folder")
    print("  2. Check problematic photos and consider replacing them")
    print("  3. Ensure each person has at least 5 high-quality photos")
    print("  4. Re-run this tool after making improvements")
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[CANCELLED] Inspection cancelled by user.")
    except Exception as e:
        logger.error(f"Inspection failed: {e}", exc_info=True)
        print(f"\n[ERROR] Inspection failed: {e}")
        print("Check logs for details.")
