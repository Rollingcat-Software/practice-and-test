"""
Fix Image Names Tool
Automatically rename all images in person folders to follow naming convention.

This will rename:
  DSC_8476.jpg → img_001.jpg
  h03.jpg → img_002.jpg
  3.jpg → img_003.jpg
  ...etc
"""

from pathlib import Path

from src.config import format_image_id, IMAGE_EXTENSIONS, is_valid_image_file


def fix_image_names(images_dir="images", dry_run=True):
    """
    Rename all images in person folders to follow naming convention.

    Args:
        images_dir: Root images directory
        dry_run: If True, only show what would be renamed (no actual changes)
    """
    images_path = Path(images_dir)

    if not images_path.exists():
        print(f"[X] Directory not found: {images_dir}")
        return

    print("=" * 70)
    print("  IMAGE NAME FIXER")
    print("  Standardizes all image names to img_XXX.jpg format")
    print("=" * 70)

    if dry_run:
        print("\n[*] DRY RUN MODE - No files will be changed")
        print("[*] Run with dry_run=False to apply changes\n")
    else:
        print("\n[!] LIVE MODE - Files will be renamed!\n")

    # Process each person folder
    for person_folder in sorted(images_path.iterdir()):
        if not person_folder.is_dir():
            continue

        # Get all image files
        image_files = []
        for file in person_folder.iterdir():
            if file.is_file():
                # Check if it's an image
                ext = file.suffix.lower()
                if ext in IMAGE_EXTENSIONS:
                    image_files.append(file)

        if not image_files:
            print(f"\n[SKIP] {person_folder.name}: No images")
            continue

        # Sort files by name for consistent ordering
        image_files.sort(key=lambda f: f.name)

        print(f"\n[{person_folder.name}] - {len(image_files)} images")
        print("-" * 70)

        # Check which ones need renaming
        needs_rename = []
        for img_file in image_files:
            if not is_valid_image_file(img_file.name):
                needs_rename.append(img_file)

        if not needs_rename:
            print("  [OK] All images already follow naming convention")
            continue

        # Rename files
        renamed_count = 0
        for img_num, old_file in enumerate(image_files, start=1):
            extension = old_file.suffix
            new_name = f"{format_image_id(img_num)}{extension}"
            new_path = old_file.parent / new_name

            # Skip if already correct
            if old_file.name == new_name:
                print(f"  [OK] {old_file.name} (already correct)")
                continue

            # Check if destination exists
            if new_path.exists():
                print(f"  [!!] {old_file.name} -> {new_name} (destination exists, skipping)")
                continue

            if dry_run:
                print(f"  [->] {old_file.name} -> {new_name}")
            else:
                old_file.rename(new_path)
                renamed_count += 1
                print(f"  [OK] {old_file.name} -> {new_name}")

        if not dry_run and renamed_count > 0:
            print(f"\n  [OK] Renamed {renamed_count} files in {person_folder.name}")


def clean_empty_folders(images_dir="images"):
    """Remove empty person folders."""
    images_path = Path(images_dir)

    print("\n" + "=" * 70)
    print("  CLEANING EMPTY FOLDERS")
    print("=" * 70 + "\n")

    removed = 0
    for person_folder in images_path.iterdir():
        if not person_folder.is_dir():
            continue

        # Check if empty
        files = list(person_folder.iterdir())
        if not files:
            print(f"[X] Removing empty folder: {person_folder.name}")
            person_folder.rmdir()
            removed += 1

    if removed == 0:
        print("[OK] No empty folders found")
    else:
        print(f"\n[OK] Removed {removed} empty folder(s)")


def check_duplicates(images_dir="images"):
    """Check for duplicate image names across different persons."""
    images_path = Path(images_dir)

    print("\n" + "=" * 70)
    print("  CHECKING FOR DUPLICATE FILES")
    print("=" * 70 + "\n")

    # Build map of filename -> list of locations
    file_map = {}

    for person_folder in sorted(images_path.iterdir()):
        if not person_folder.is_dir():
            continue

        for file in person_folder.iterdir():
            if file.is_file():
                ext = file.suffix.lower()
                if ext in IMAGE_EXTENSIONS:
                    if file.name not in file_map:
                        file_map[file.name] = []
                    file_map[file.name].append(person_folder.name)

    # Find duplicates
    duplicates = {name: locations for name, locations in file_map.items() if len(locations) > 1}

    if not duplicates:
        print("[OK] No duplicate filenames found")
    else:
        print("[!] Found duplicate filenames (same name in multiple person folders):\n")
        for filename, locations in duplicates.items():
            print(f"  {filename}:")
            for loc in locations:
                print(f"    - {loc}/")

        print("\n[!] WARNING: These might be:")
        print("    1. Same person (both folders should be merged)")
        print("    2. Different people with same original filename")
        print("    After renaming, duplicates will be resolved automatically")


if __name__ == "__main__":

    print("\n" + "=" * 70)
    print("  IMAGE NAME STANDARDIZATION TOOL")
    print("=" * 70)

    # Check for duplicates
    check_duplicates()

    # Show what would be renamed (dry run)
    print("\n")
    fix_image_names(dry_run=True)

    # Clean empty folders
    clean_empty_folders()

    # Ask for confirmation
    print("\n" + "=" * 70)
    response = input("\nApply these changes? (yes/no): ").strip().lower()

    if response in ['yes', 'y']:
        print("\n[*] Applying changes...\n")
        fix_image_names(dry_run=False)
        clean_empty_folders()

        print("\n" + "=" * 70)
        print("[OK] Image names have been standardized!")
        print("=" * 70)

        # Show final structure
        print("\nFinal structure:")
        from src.services import PersonManager

        manager = PersonManager()
        persons = manager.scan_all_persons()

        for person in persons:
            print(f"\n  {person.folder_name}/ ({person.image_count} images)")
            for img_path in person.image_paths[:5]:
                print(f"    - {Path(img_path).name}")
            if person.image_count > 5:
                print(f"    ... and {person.image_count - 5} more")

    else:
        print("\n[X] Changes cancelled. No files were modified.")
