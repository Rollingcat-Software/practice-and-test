"""
Migration Script: Reorganize Images to New Structure

This script converts old image structure to the new person-based structure.

Old structure:
  images/
    kisi_A_1.jpg
    kisi_A_2.jpg
    kisi_B_1.jpg

New structure:
  images/
    person_0001/
      img_001.jpg
      img_002.jpg
    person_0002/
      img_001.jpg
"""

import shutil
from collections import defaultdict
from pathlib import Path

from src.config import format_person_id, format_image_id, IMAGE_EXTENSIONS


def parse_old_filename(filename: str) -> tuple:
    """
    Parse old filename format.

    Old format: kisi_A_1.jpg, kisi_B_1.jpg, etc.
    Returns: (person_letter, image_num)
    """
    name_without_ext = filename.rsplit('.', 1)[0]
    parts = name_without_ext.split('_')

    if len(parts) >= 3 and parts[0] == 'kisi':
        person_letter = parts[1]  # A, B, C, etc.
        try:
            image_num = int(parts[2])
            return (person_letter, image_num)
        except ValueError:
            pass

    return None


def migrate_images(images_dir: str = "images", backup: bool = True):
    """
    Migrate images from old structure to new structure.

    Args:
        images_dir: Root images directory
        backup: If True, creates backup before migration
    """
    images_path = Path(images_dir)

    if not images_path.exists():
        print(f"[X] Images directory not found: {images_dir}")
        return

    # Step 1: Find all image files in root directory
    print("[*] Scanning for images to migrate...")
    old_images = []

    for file in images_path.iterdir():
        if file.is_file() and any(file.name.lower().endswith(ext) for ext in IMAGE_EXTENSIONS):
            old_images.append(file)

    if not old_images:
        print("[OK] No old-format images found. Directory is already organized!")
        return

    print(f"[*] Found {len(old_images)} images to migrate")

    # Step 2: Create backup if requested
    if backup:
        backup_dir = images_path.parent / "images_backup"
        print(f"[*] Creating backup at: {backup_dir}")
        if backup_dir.exists():
            shutil.rmtree(backup_dir)
        shutil.copytree(images_path, backup_dir)
        print("[OK] Backup created successfully")

    # Step 3: Group images by person
    person_groups = defaultdict(list)

    for img_file in old_images:
        result = parse_old_filename(img_file.name)

        if result:
            person_letter, image_num = result
            person_groups[person_letter].append((img_file, image_num))
        else:
            print(f"[!] Skipping unrecognized filename: {img_file.name}")

    # Step 4: Create person folders and migrate images
    print(f"\n[*] Migrating images into {len(person_groups)} person folders...")

    # Map letters to numbers (A=1, B=2, etc.)
    person_mapping = {}
    for person_id, (person_letter, images) in enumerate(sorted(person_groups.items()), start=1):
        person_mapping[person_letter] = person_id

        # Create person folder
        person_folder_name = format_person_id(person_id)
        person_folder = images_path / person_folder_name
        person_folder.mkdir(exist_ok=True)

        print(f"\n  [DIR] {person_letter} -> {person_folder_name}")

        # Move images to person folder
        for old_file, old_img_num in sorted(images, key=lambda x: x[1]):
            # Get file extension
            extension = old_file.suffix

            # Create new filename
            new_filename = f"{format_image_id(old_img_num)}{extension}"
            new_path = person_folder / new_filename

            # Move file
            shutil.move(str(old_file), str(new_path))
            print(f"     [+] {old_file.name} -> {new_filename}")

    # Step 5: Save migration mapping
    mapping_file = images_path.parent / "database" / "migration_mapping.json"
    mapping_file.parent.mkdir(exist_ok=True)

    import json
    mapping_data = {
        "migration_date": str(Path(__file__).stat().st_mtime),
        "person_mapping": {
            f"kisi_{letter}": {
                "person_id": pid,
                "folder_name": format_person_id(pid)
            }
            for letter, pid in person_mapping.items()
        }
    }

    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(mapping_data, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] Migration completed successfully!")
    print(f"   - Migrated {len(old_images)} images")
    print(f"   - Created {len(person_groups)} person folders")
    print(f"   - Mapping saved to: {mapping_file}")

    if backup:
        print(f"   - Backup available at: {backup_dir}")


def show_structure(images_dir: str = "images"):
    """Display the current directory structure."""
    images_path = Path(images_dir)

    if not images_path.exists():
        print(f"[X] Directory not found: {images_dir}")
        return

    print(f"\nDirectory Structure: {images_dir}/")
    print("=" * 60)

    # Count persons and images
    person_count = 0
    total_images = 0

    for item in sorted(images_path.iterdir()):
        if item.is_dir():
            person_count += 1
            images = [f for f in item.iterdir() if f.is_file()]
            total_images += len(images)

            print(f"\n  [DIR] {item.name}/ ({len(images)} images)")
            for img in sorted(images)[:5]:  # Show first 5
                print(f"     - {img.name}")
            if len(images) > 5:
                print(f"     ... and {len(images) - 5} more")

        elif item.is_file():
            # File in root (old format)
            print(f"  [FILE] {item.name} (needs migration)")

    print("\n" + "=" * 60)
    print(f"Total: {person_count} persons, {total_images} images")


if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("  IMAGE MIGRATION TOOL")
    print("  Converts old structure to person-based folders")
    print("=" * 60)

    # Show current structure
    show_structure()

    # Ask for confirmation
    print("\n[!] This will reorganize your images directory.")
    print("   A backup will be created at: images_backup/")
    response = input("\nProceed with migration? (yes/no): ").strip().lower()

    if response in ['yes', 'y']:
        migrate_images(backup=True)
        print("\n" + "=" * 60)
        show_structure()
    else:
        print("\n[X] Migration cancelled.")
