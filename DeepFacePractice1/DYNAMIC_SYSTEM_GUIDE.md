# Dynamic Multi-Person Recognition System

## Overview

Your DeepFace Practice Project now supports **dynamic, scalable person management** that can handle unlimited persons
with professional naming conventions.

## New Folder Structure

```
images/
├── person_0001/          # Person 1
│   ├── img_001.jpg       # First image
│   ├── img_002.jpg       # Second image
│   └── img_003.jpg       # Third image
├── person_0002/          # Person 2
│   └── img_001.jpg
├── person_0003/          # Person 3
│   ├── img_001.jpg
│   └── img_002.jpg
...
└── person_9999/          # Can scale to 9,999 people!
    └── img_001.jpg
```

## Naming Convention

### Person Folders

- **Format**: `person_XXXX` (4 digits)
- **Examples**: `person_0001`, `person_0816`, `person_9999`
- **Capacity**: Up to 9,999 persons
- **Benefits**:
    - Proper alphabetical sorting
    - No sorting issues (unlike person1, person10, person2)
    - Professional standard (matches academic datasets)
    - Easy to scale

### Image Files

- **Format**: `img_XXX.jpg` (3 digits)
- **Examples**: `img_001.jpg`, `img_100.jpg`, `img_999.jpg`
- **Capacity**: Up to 999 images per person
- **Supported formats**: `.jpg`, `.jpeg`, `.png`, `.bmp`

## How to Add New Persons

### Method 1: Manual (Recommended for beginners)

1. Create a new folder in `images/` directory:
   ```
   images/person_0003/
   ```

2. Add images to the folder:
   ```
   images/person_0003/img_001.jpg
   images/person_0003/img_002.jpg
   ```

3. The system automatically detects new persons!

### Method 2: Programmatic (Advanced)

```python
from src.services import PersonManager

manager = PersonManager()

# Create new person folder
person = manager.create_person(person_id=3, name="John Doe")

# Add images
manager.add_image_to_person(
    person_id=3,
    source_image_path="path/to/photo.jpg"
)
```

## Migration Tool

Already have images in the old format? Use the migration tool:

```bash
python migrate_images.py
```

This will:

- ✅ Automatically organize your images into person folders
- ✅ Create a backup at `images_backup/`
- ✅ Rename files to follow the naming convention
- ✅ Generate a migration mapping file

## Key Features

### 1. Automatic Person Discovery

```python
from src.services import PersonManager

manager = PersonManager()
persons = manager.scan_all_persons()

print(f"Found {len(persons)} persons")
for person in persons:
    print(f"{person.folder_name}: {person.image_count} images")
```

### 2. Dynamic Scaling

- Add persons without changing any code
- System automatically detects new folders
- No hardcoded image paths
- Works with 1 person or 1,000 persons

### 3. Professional Organization

- Industry-standard naming (matches VGGFace2, LFW datasets)
- Zero-padded numbers for proper sorting
- Self-documenting folder structure
- Easy to maintain and extend

## Configuration

Edit `src/config/naming_config.py` to customize:

```python
# Change number of digits for persons
PERSON_ID_DIGITS = 6  # Now supports 999,999 persons!

# Change number of digits for images
IMAGE_ID_DIGITS = 4   # Now supports 9,999 images per person!

# Change prefixes
PERSON_PREFIX = "person_"
IMAGE_PREFIX = "img_"
```

## New Demo

**Demo 4: Dynamic Multi-Person Recognition**

Run:

```bash
python quick_start.py
# Select option 4

# Or run directly:
python src/demos/demo_4_dynamic_recognition.py
```

Features:

- Automatic person discovery
- Cross-person verification matrix
- Intra-person verification (multiple images of same person)
- Person identification from query image
- Scalability demonstration

## Use Cases

### Small Projects (2-10 people)

Perfect for:

- Family photo organization
- Personal projects
- Learning face recognition

### Medium Projects (10-100 people)

Good for:

- Small business attendance systems
- Event photo organization
- Academic research

### Large Projects (100-9,999 people)

Suitable for:

- Organization-wide systems
- Research datasets
- Production applications

## Advanced Usage

### Finding a Specific Person

```python
manager = PersonManager()
person = manager.get_person(person_id=1)
print(f"Person 1 has {person.image_count} images")
```

### Getting Random Persons for Testing

```python
random_persons = manager.get_random_persons(count=5)
```

### Cross-Person Pairs

```python
pairs = manager.get_person_pairs()
# Returns all possible pairs of different persons
```

### Person Registry (Metadata)

```python
persons = manager.scan_all_persons()

# Add metadata
for person in persons:
    person.name = f"Person {person.person_id}"
    person.metadata = {"age": 25, "department": "Engineering"}

# Save registry
manager.save_registry(persons)

# Load registry
persons = manager.load_registry()
```

## Best Practices

1. **Consistent Naming**: Always use the naming convention
2. **One Person Per Folder**: Don't mix different people in one folder
3. **Clear Images**: Use good quality face photos
4. **Multiple Images**: Add 2-5 images per person for better accuracy
5. **Backup**: The migration tool creates backups automatically

## File Organization

```
DeepFacePractice1/
├── images/                     # Person folders
│   ├── person_0001/
│   ├── person_0002/
│   └── ...
├── database/                   # Person registry & metadata
│   ├── person_registry.json
│   └── migration_mapping.json
├── src/
│   ├── config/
│   │   └── naming_config.py   # Naming convention settings
│   ├── services/
│   │   └── person_manager.py  # Person management service
│   └── demos/
│       └── demo_4_dynamic_recognition.py
└── migrate_images.py          # Migration tool
```

## Troubleshooting

### "No persons found"

- Check if folders follow `person_XXXX` format
- Verify images are in correct format (`.jpg`, `.png`, etc.)
- Run `python migrate_images.py` if you have old-format images

### "Person X already exists"

- Check if folder `person_000X` already exists
- Use different person ID

### Images not detected

- Check file extensions are supported
- Verify naming: `img_001.jpg`, not `image_001.jpg`
- Files must follow `img_XXX.ext` format

## Next Steps

1. ✅ Add more person folders to test scalability
2. ✅ Try Demo 4 to see the system in action
3. ✅ Customize configuration for your needs
4. ✅ Build your own face recognition application!

## Support

- Check `README.md` for general project information
- See `PROJECT_SUMMARY.md` for architecture overview
- Run `python quick_start.py` to access all tutorials
