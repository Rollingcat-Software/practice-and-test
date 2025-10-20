"""
Person Manager Service
Handles dynamic person folder management and image organization.

Key Features:
- Automatic person folder discovery
- Scalable to unlimited persons
- Image enumeration per person
- Person metadata management
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from ..config import (
    format_person_id,
    format_image_id,
    parse_person_id,
    is_valid_person_folder,
    is_valid_image_file,
    IMAGES_ROOT_DIR,
    PERSON_REGISTRY_FILE,
)


@dataclass
class Person:
    """
    Represents a person in the database.

    Attributes:
        person_id: Unique numeric ID (e.g., 1, 2, 816)
        folder_name: Standardized folder name (e.g., "person_0001")
        name: Optional human-readable name
        image_count: Number of images for this person
        image_paths: List of absolute paths to images
        metadata: Additional custom metadata
    """
    person_id: int
    folder_name: str
    name: Optional[str] = None
    image_count: int = 0
    image_paths: List[str] = None
    metadata: Dict = None

    def __post_init__(self):
        if self.image_paths is None:
            self.image_paths = []
        if self.metadata is None:
            self.metadata = {}


class PersonManager:
    """
    Service for managing persons and their images dynamically.

    This class implements:
    - Repository Pattern: Central data access for persons
    - Single Responsibility: Only handles person/image management
    - Open/Closed: Easy to extend with new features
    """

    def __init__(self, images_root: str = IMAGES_ROOT_DIR):
        """
        Initialize the PersonManager.

        Args:
            images_root: Root directory containing person folders
        """
        self.images_root = Path(images_root)
        self.registry_file = Path(PERSON_REGISTRY_FILE)
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        self.images_root.mkdir(parents=True, exist_ok=True)
        self.registry_file.parent.mkdir(parents=True, exist_ok=True)

    def scan_all_persons(self) -> List[Person]:
        """
        Scan the images directory and discover all persons.

        Returns:
            List of Person objects found in the directory

        Example:
            >>> manager = PersonManager()
            >>> persons = manager.scan_all_persons()
            >>> print(f"Found {len(persons)} people")
        """
        persons = []

        if not self.images_root.exists():
            return persons

        # Scan all subdirectories
        for folder in sorted(self.images_root.iterdir()):
            if not folder.is_dir():
                continue

            folder_name = folder.name

            # Check if it follows our naming convention
            if not is_valid_person_folder(folder_name):
                continue

            # Extract person ID
            person_id = parse_person_id(folder_name)

            # Find all images for this person
            image_paths = self._find_images_in_folder(folder)

            # Create Person object
            person = Person(
                person_id=person_id,
                folder_name=folder_name,
                image_count=len(image_paths),
                image_paths=image_paths
            )

            persons.append(person)

        return persons

    def _find_images_in_folder(self, folder: Path) -> List[str]:
        """
        Find all valid images in a person's folder.

        Args:
            folder: Path to person's folder

        Returns:
            List of absolute paths to image files
        """
        images = []

        for file in sorted(folder.iterdir()):
            if not file.is_file():
                continue

            if is_valid_image_file(file.name):
                images.append(str(file.absolute()))

        return images

    def get_person(self, person_id: int) -> Optional[Person]:
        """
        Get a specific person by ID.

        Args:
            person_id: Numeric person ID

        Returns:
            Person object if found, None otherwise
        """
        folder_name = format_person_id(person_id)
        folder_path = self.images_root / folder_name

        if not folder_path.exists():
            return None

        image_paths = self._find_images_in_folder(folder_path)

        return Person(
            person_id=person_id,
            folder_name=folder_name,
            image_count=len(image_paths),
            image_paths=image_paths
        )

    def create_person(self, person_id: int, name: Optional[str] = None) -> Person:
        """
        Create a new person folder.

        Args:
            person_id: Numeric person ID
            name: Optional human-readable name

        Returns:
            Created Person object

        Raises:
            ValueError: If person already exists
        """
        folder_name = format_person_id(person_id)
        folder_path = self.images_root / folder_name

        if folder_path.exists():
            raise ValueError(f"Person {person_id} already exists at {folder_path}")

        folder_path.mkdir(parents=True, exist_ok=True)

        person = Person(
            person_id=person_id,
            folder_name=folder_name,
            name=name,
            image_count=0,
            image_paths=[]
        )

        return person

    def add_image_to_person(
            self,
            person_id: int,
            source_image_path: str,
            image_num: Optional[int] = None
    ) -> str:
        """
        Add an image to a person's folder.

        Args:
            person_id: Person's numeric ID
            source_image_path: Path to source image file
            image_num: Optional specific image number (auto-increments if None)

        Returns:
            Path to the copied image

        Raises:
            ValueError: If person doesn't exist or source image not found
        """
        person = self.get_person(person_id)
        if person is None:
            raise ValueError(f"Person {person_id} not found")

        source_path = Path(source_image_path)
        if not source_path.exists():
            raise ValueError(f"Source image not found: {source_image_path}")

        # Determine image number
        if image_num is None:
            image_num = person.image_count + 1

        # Get file extension
        extension = source_path.suffix

        # Create destination filename
        dest_filename = f"{format_image_id(image_num)}{extension}"
        dest_path = self.images_root / person.folder_name / dest_filename

        # Copy image
        import shutil
        shutil.copy2(source_path, dest_path)

        return str(dest_path)

    def get_all_person_ids(self) -> List[int]:
        """
        Get list of all person IDs.

        Returns:
            Sorted list of person IDs
        """
        persons = self.scan_all_persons()
        return sorted([p.person_id for p in persons])

    def get_person_count(self) -> int:
        """Get total number of persons."""
        return len(self.get_all_person_ids())

    def get_random_persons(self, count: int = 2) -> List[Person]:
        """
        Get random persons for testing/demos.

        Args:
            count: Number of random persons to retrieve

        Returns:
            List of Person objects
        """
        import random
        persons = self.scan_all_persons()
        return random.sample(persons, min(count, len(persons)))

    def get_person_pairs(self) -> List[Tuple[Person, Person]]:
        """
        Get all possible pairs of different persons.

        Useful for verification testing (comparing different people).

        Returns:
            List of (person1, person2) tuples
        """
        persons = self.scan_all_persons()
        pairs = []

        for i, p1 in enumerate(persons):
            for p2 in persons[i + 1:]:
                pairs.append((p1, p2))

        return pairs

    def save_registry(self, persons: List[Person]) -> None:
        """
        Save person registry to JSON file.

        Args:
            persons: List of Person objects to save
        """
        data = {
            "version": "1.0",
            "person_count": len(persons),
            "persons": [
                {
                    "person_id": p.person_id,
                    "folder_name": p.folder_name,
                    "name": p.name,
                    "image_count": p.image_count,
                    "metadata": p.metadata
                }
                for p in persons
            ]
        }

        with open(self.registry_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_registry(self) -> List[Person]:
        """
        Load person registry from JSON file.

        Returns:
            List of Person objects
        """
        if not self.registry_file.exists():
            return []

        with open(self.registry_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        persons = []
        for p_data in data.get("persons", []):
            person = self.get_person(p_data["person_id"])
            if person:
                person.name = p_data.get("name")
                person.metadata = p_data.get("metadata", {})
                persons.append(person)

        return persons

    def __str__(self) -> str:
        """Human-readable representation."""
        count = self.get_person_count()
        return f"PersonManager(persons={count}, root={self.images_root})"
