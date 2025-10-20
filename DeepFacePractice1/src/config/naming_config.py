"""
Naming Convention Configuration
Centralized configuration for person and image naming standards.
"""

# Person Folder Naming
PERSON_PREFIX = "person_"
PERSON_ID_DIGITS = 4  # Supports up to 9,999 people

# Image File Naming
IMAGE_PREFIX = "img_"
IMAGE_ID_DIGITS = 3  # Supports up to 999 images per person
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]

# Directories
IMAGES_ROOT_DIR = "images"
DATABASE_DIR = "database"
OUTPUT_DIR = "output"

# Person Registry
PERSON_REGISTRY_FILE = "database/person_registry.json"


def format_person_id(person_num: int) -> str:
    """
    Format a person number into standardized folder name.

    Args:
        person_num: Person number (1, 2, 3, ..., 816, etc.)

    Returns:
        Formatted person ID (e.g., "person_0001", "person_0816")

    Examples:
        >>> format_person_id(1)
        'person_0001'
        >>> format_person_id(816)
        'person_0816'
    """
    return f"{PERSON_PREFIX}{person_num:0{PERSON_ID_DIGITS}d}"


def format_image_id(image_num: int) -> str:
    """
    Format an image number into standardized filename (without extension).

    Args:
        image_num: Image number (1, 2, 3, ..., 100, etc.)

    Returns:
        Formatted image ID (e.g., "img_001", "img_100")

    Examples:
        >>> format_image_id(1)
        'img_001'
        >>> format_image_id(100)
        'img_100'
    """
    return f"{IMAGE_PREFIX}{image_num:0{IMAGE_ID_DIGITS}d}"


def parse_person_id(folder_name: str) -> int:
    """
    Extract person number from folder name.

    Args:
        folder_name: Folder name (e.g., "person_0001")

    Returns:
        Person number as integer

    Examples:
        >>> parse_person_id("person_0001")
        1
        >>> parse_person_id("person_0816")
        816
    """
    if not folder_name.startswith(PERSON_PREFIX):
        raise ValueError(f"Invalid person folder name: {folder_name}")

    try:
        return int(folder_name[len(PERSON_PREFIX):])
    except ValueError:
        raise ValueError(f"Invalid person folder name: {folder_name}")


def parse_image_id(filename: str) -> int:
    """
    Extract image number from filename.

    Args:
        filename: Image filename (e.g., "img_001.jpg")

    Returns:
        Image number as integer

    Examples:
        >>> parse_image_id("img_001.jpg")
        1
        >>> parse_image_id("img_100.jpg")
        100
    """
    # Remove extension
    name_without_ext = filename.rsplit('.', 1)[0]

    if not name_without_ext.startswith(IMAGE_PREFIX):
        raise ValueError(f"Invalid image filename: {filename}")

    try:
        return int(name_without_ext[len(IMAGE_PREFIX):])
    except ValueError:
        raise ValueError(f"Invalid image filename: {filename}")


def is_valid_person_folder(folder_name: str) -> bool:
    """Check if folder name follows person naming convention."""
    try:
        parse_person_id(folder_name)
        return True
    except ValueError:
        return False


def is_valid_image_file(filename: str) -> bool:
    """Check if filename follows image naming convention."""
    try:
        # Check extension
        has_valid_ext = any(filename.lower().endswith(ext) for ext in IMAGE_EXTENSIONS)
        if not has_valid_ext:
            return False

        # Check naming pattern
        parse_image_id(filename)
        return True
    except ValueError:
        return False
