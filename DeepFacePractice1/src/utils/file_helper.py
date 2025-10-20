"""
File Helper Utility
Helps with file operations and path management.

Educational Notes:
- Encapsulates file operations (Single Responsibility)
- Reusable across the application (DRY)
- Makes path handling safer and easier
"""

import shutil
from pathlib import Path
from typing import List, Optional


class FileHelper:
    """
    Helper class for file operations.

    Demonstrates:
    - Static methods for utility functions
    - Path validation and safety
    """

    # Supported image extensions
    IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}

    @staticmethod
    def ensure_directory(directory: str) -> Path:
        """
        Ensure a directory exists, create if it doesn't.

        Args:
            directory: Directory path

        Returns:
            Path object for the directory
        """
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @staticmethod
    def is_image_file(file_path: str) -> bool:
        """
        Check if a file is an image based on extension.

        Args:
            file_path: Path to the file

        Returns:
            True if file appears to be an image
        """
        return Path(file_path).suffix.lower() in FileHelper.IMAGE_EXTENSIONS

    @staticmethod
    def find_images(directory: str, recursive: bool = True) -> List[str]:
        """
        Find all image files in a directory.

        Args:
            directory: Directory to search
            recursive: If True, search subdirectories

        Returns:
            List of image file paths (as strings)
        """
        dir_path = Path(directory)

        if not dir_path.exists():
            raise ValueError(f"Directory not found: {directory}")

        if recursive:
            # Search recursively
            image_files = [
                str(f) for f in dir_path.rglob('*')
                if f.suffix.lower() in FileHelper.IMAGE_EXTENSIONS
            ]
        else:
            # Search only in top level
            image_files = [
                str(f) for f in dir_path.glob('*')
                if f.suffix.lower() in FileHelper.IMAGE_EXTENSIONS
            ]

        return sorted(image_files)

    @staticmethod
    def validate_image_path(img_path: str) -> Path:
        """
        Validate that an image path exists and is readable.

        Args:
            img_path: Path to validate

        Returns:
            Path object if valid

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is not an image
        """
        path = Path(img_path)

        if not path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        if not FileHelper.is_image_file(img_path):
            raise ValueError(f"Not an image file: {img_path}")

        return path

    @staticmethod
    def copy_file(source: str, destination: str, overwrite: bool = False) -> Path:
        """
        Copy a file to a new location.

        Args:
            source: Source file path
            destination: Destination file path
            overwrite: If True, overwrite existing file

        Returns:
            Path to destination file

        Raises:
            FileExistsError: If destination exists and overwrite=False
        """
        src_path = Path(source)
        dst_path = Path(destination)

        if not src_path.exists():
            raise FileNotFoundError(f"Source file not found: {source}")

        if dst_path.exists() and not overwrite:
            raise FileExistsError(f"Destination already exists: {destination}")

        # Ensure destination directory exists
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(src_path, dst_path)
        return dst_path

    @staticmethod
    def get_relative_path(file_path: str, base_dir: str) -> str:
        """
        Get relative path from a base directory.

        Args:
            file_path: Full file path
            base_dir: Base directory

        Returns:
            Relative path as string
        """
        file_p = Path(file_path)
        base_p = Path(base_dir)

        try:
            return str(file_p.relative_to(base_p))
        except ValueError:
            # Not relative, return absolute
            return str(file_p.absolute())

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """
        Remove invalid characters from filename.

        Args:
            filename: Original filename

        Returns:
            Sanitized filename safe for all systems
        """
        # Remove or replace invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')

        return filename

    @staticmethod
    def generate_output_path(
            input_path: str,
            output_dir: str,
            suffix: str = "_output",
            extension: Optional[str] = None
    ) -> Path:
        """
        Generate output path based on input file.

        Args:
            input_path: Original input file path
            output_dir: Output directory
            suffix: Suffix to add to filename
            extension: New extension (None to keep original)

        Returns:
            Path for output file

        Example:
            input_path = "images/face.jpg"
            output_dir = "output"
            suffix = "_analyzed"
            => output/face_analyzed.jpg
        """
        input_p = Path(input_path)

        # Use original extension or provided one
        if extension is None:
            extension = input_p.suffix
        elif not extension.startswith('.'):
            extension = f'.{extension}'

        # Create new filename
        new_filename = f"{input_p.stem}{suffix}{extension}"

        # Build output path
        output_path = Path(output_dir) / new_filename

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        return output_path
