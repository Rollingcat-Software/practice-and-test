"""
Demo 3: Face Embeddings & Recognition
Learn about face embeddings and how to use them for recognition.

Educational Focus:
- Understanding face embeddings (vector representations)
- Comparing embeddings mathematically
- Building a simple face recognition system
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.services.face_recognition_service import FaceRecognitionService
from src.services.person_manager import PersonManager
from src.utils.visualizer import FaceVisualizer
from src.utils.logger import setup_logger

logger = setup_logger("Demo3_Embeddings")
visualizer = FaceVisualizer()
person_manager = PersonManager()


def demo_extract_embeddings():
    """
    Extract and examine face embeddings.

    Learn: What embeddings are and what they look like.
    """
    print("\n" + "=" * 60)
    print("DEMO 3A: Extracting Face Embeddings")
    print("=" * 60 + "\n")

    service = FaceRecognitionService(model_name="Facenet512")

    # Get a photo from the database
    persons = person_manager.scan_all_persons()
    if not persons or not persons[0].image_paths:
        print("[WARN] No images found in database. Please add images to 'images/person_XXXX/' folders.")
        return

    img_path = persons[0].image_paths[0]
    print(f"Using image from database: {Path(img_path).name} (person: {persons[0].folder_name})\n")

    logger.info(f"Extracting embedding from: {img_path}")

    try:
        embedding = service.extract_embedding(img_path)

        print(embedding)

        print("\n" + "=" * 60)
        print("WHAT IS AN EMBEDDING?")
        print("=" * 60)
        print("""
An embedding is a numerical representation of a face:
- Converts a face image into a vector of numbers
- Similar faces have similar vectors
- Different faces have different vectors
- Enables mathematical comparison of faces

Think of it like a "fingerprint" for a face, but in number form.
        """)

        stats = embedding.statistics()
        print("\nEmbedding Statistics:")
        print(f"  Dimension: {embedding.dimension}")
        print(f"  Min value: {stats['min']:.6f}")
        print(f"  Max value: {stats['max']:.6f}")
        print(f"  Mean: {stats['mean']:.6f}")
        print(f"  Std Dev: {stats['std']:.6f}")

        # Show a sample of the embedding
        print(f"\nFirst 20 values of the embedding:")
        print(embedding.embedding[:20])

        print(f"\nKey Point:")
        print(f"  This {embedding.dimension}-dimensional vector uniquely represents the face.")
        print(f"  We can compare vectors to measure face similarity!")

    except Exception as e:
        logger.error(f"Embedding extraction failed: {e}")


def demo_compare_embeddings():
    """
    Compare embeddings mathematically.

    Learn: How to measure similarity between faces using embeddings.
    """
    print("\n" + "=" * 60)
    print("DEMO 3B: Comparing Embeddings")
    print("=" * 60 + "\n")

    service = FaceRecognitionService(model_name="Facenet512")

    # Get images from the database
    persons = person_manager.scan_all_persons()
    existing_images = {}

    # Get 2 images from first person (same person comparison)
    if persons and len(persons[0].image_paths) >= 2:
        existing_images[f"{persons[0].folder_name} - Photo 1"] = persons[0].image_paths[0]
        existing_images[f"{persons[0].folder_name} - Photo 2"] = persons[0].image_paths[1]

    # Get 1 image from second person (different person comparison)
    if len(persons) > 1 and persons[1].image_paths:
        existing_images[f"{persons[1].folder_name} - Photo 1"] = persons[1].image_paths[0]

    if len(existing_images) < 2:
        print("[WARN] Need at least 2 images from database for comparison")
        print("Please add more images to person folders.")
        return

    print(f"Comparing {len(existing_images)} images from database\n")

    logger.info("Extracting embeddings from all images...")

    embeddings = {}
    for name, img_path in existing_images.items():
        try:
            emb = service.extract_embedding(img_path)
            embeddings[name] = emb
            print(f"[OK] Extracted embedding from: {name}")
        except Exception as e:
            print(f"[FAIL] Failed for {name}: {e}")

    if len(embeddings) < 2:
        print("Not enough embeddings for comparison")
        return

    # Compare all pairs
    print("\n" + "=" * 60)
    print("PAIRWISE COMPARISONS")
    print("=" * 60)

    names = list(embeddings.keys())

    print("\nCosine Similarity (higher = more similar, range: -1 to 1):")
    print(f"{'Comparison':<50} {'Similarity':<12} {'Interpretation'}")
    print("-" * 80)

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            name1, name2 = names[i], names[j]
            emb1, emb2 = embeddings[name1], embeddings[name2]

            similarity = emb1.cosine_similarity(emb2)

            # Interpret similarity
            if similarity > 0.7:
                interpretation = "Very similar (likely same person)"
            elif similarity > 0.5:
                interpretation = "Somewhat similar"
            elif similarity > 0.3:
                interpretation = "Slightly similar"
            else:
                interpretation = "Not similar (different people)"

            comparison_str = f"{name1} vs {name2}"
            print(f"{comparison_str:<50} {similarity:<12.4f} {interpretation}")

    print("\n\nEuclidean Distance (lower = more similar):")
    print(f"{'Comparison':<50} {'Distance':<12} {'Interpretation'}")
    print("-" * 80)

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            name1, name2 = names[i], names[j]
            emb1, emb2 = embeddings[name1], embeddings[name2]

            distance = emb1.euclidean_distance(emb2)

            # Interpret distance (thresholds vary by model)
            if distance < 10:
                interpretation = "Very similar (likely same person)"
            elif distance < 20:
                interpretation = "Somewhat similar"
            elif distance < 30:
                interpretation = "Slightly similar"
            else:
                interpretation = "Not similar (different people)"

            comparison_str = f"{name1} vs {name2}"
            print(f"{comparison_str:<50} {distance:<12.4f} {interpretation}")

    # Visualize
    if len(embeddings) >= 2:
        visualizer.visualize_embedding_comparison(
            embeddings,
            save_path="output/embedding_comparison.png",
            show=False
        )
        print("\n[OK] Visualization saved to: output/embedding_comparison.png")


def demo_model_embeddings():
    """
    Compare embeddings from different models.

    Learn: How different models produce different embedding dimensions.
    """
    print("\n" + "=" * 60)
    print("DEMO 3C: Embeddings from Different Models")
    print("=" * 60 + "\n")

    # Get a photo from the database
    persons = person_manager.scan_all_persons()
    if not persons or not persons[0].image_paths:
        print("[WARN] No images found in database.")
        return

    img_path = persons[0].image_paths[0]
    print(f"Using image: {Path(img_path).name}\n")

    models = [
        ("OpenFace", 128),
        ("Facenet", 128),
        ("Facenet512", 512),
        ("ArcFace", 512),
    ]

    print(f"Extracting embeddings using different models:\n")
    print(f"{'Model':<15} {'Dimension':<12} {'Sample Values'}")
    print("-" * 80)

    for model_name, expected_dim in models:
        try:
            service = FaceRecognitionService(model_name=model_name)
            embedding = service.extract_embedding(img_path)

            # Show first 5 values
            sample = ", ".join([f"{v:.3f}" for v in embedding.embedding[:5]])

            print(f"{model_name:<15} {embedding.dimension:<12} [{sample}, ...]")

        except Exception as e:
            print(f"{model_name:<15} ERROR: {e}")

    print("\nKey Observations:")
    print("  - Different models produce different dimension embeddings")
    print("  - Higher dimensions can capture more facial details")
    print("  - But they're also slower to compute and compare")
    print("  - Choose based on your accuracy/speed requirements")


def demo_face_search():
    """
    Simple face search/recognition demo.

    Learn: How to search for a face in a database.
    """
    print("\n" + "=" * 60)
    print("DEMO 3D: Face Recognition - Database Search")
    print("=" * 60 + "\n")

    print("""
CONCEPT: Face Recognition vs Verification

Verification (1:1):
  "Is this person the same as that person?"
  Example: Unlocking your phone with face

Recognition (1:N):
  "Who is this person from my database of known people?"
  Example: Tagging friends in photos

We'll demonstrate recognition by:
1. Building a database of known faces
2. Searching for a query face in that database
    """)

    # Use the existing images directory as the database
    db_path = Path("images")

    if not db_path.exists():
        print("\n[WARN]  No face database found.")
        print("Please add images to 'images/person_XXXX/' folders.")
        print("\nSkipping this demo for now.")
        return

    # Check if we have persons with images
    persons = person_manager.scan_all_persons()
    if not persons or not any(p.image_paths for p in persons):
        print("\n[WARN]  No images found in database.")
        print("Please add images to 'images/person_XXXX/' folders.")
        print("\nSkipping this demo for now.")
        return

    service = FaceRecognitionService()

    # Use first person's first image as query
    query_img = persons[0].image_paths[0]
    print(f"Query image: {Path(query_img).name} from {persons[0].folder_name}\n")

    logger.info(f"Searching for: {query_img} in database: {db_path}")

    try:
        matches = service.find_faces_in_database(
            img_path=query_img,
            db_path=str(db_path)
        )

        if matches:
            print(f"\n[OK] Found {len(matches)} match(es)!")
            print(f"\n{'Rank':<6} {'Match Image':<40} {'Distance':<12} {'Similarity'}")
            print("-" * 80)

            for i, match in enumerate(matches[:5], 1):  # Show top 5
                img_path = match.get('identity', 'Unknown')
                distance = match.get('distance', 0)

                # Normalize distance to percentage (rough approximation)
                similarity = max(0, min(100, (1 - distance / 2) * 100))

                print(f"{i:<6} {Path(img_path).name:<40} {distance:<12.4f} {similarity:.1f}%")

        else:
            print("\n[FAIL] No matches found in database")

    except Exception as e:
        logger.error(f"Database search failed: {e}")
        print(f"\nError details: {e}")


if __name__ == "__main__":
    print("\n" + " " * 30)
    print("FACE EMBEDDINGS & RECOGNITION TUTORIAL")
    print("Learn the mathematics behind face recognition")
    print(" " * 30)

    # Create directories
    Path("output").mkdir(exist_ok=True)
    Path("images").mkdir(exist_ok=True)

    # Run demos
    demo_extract_embeddings()
    demo_compare_embeddings()
    demo_model_embeddings()
    demo_face_search()

    print("\n" + "[OK] " * 30)
    print("TUTORIAL COMPLETE!")
    print("\nYou've learned:")
    print("  [OK] What face embeddings are")
    print("  [OK] How to extract embeddings from faces")
    print("  [OK] How to compare embeddings mathematically")
    print("  [OK] The difference between verification and recognition")
    print("  [OK] How to search for faces in a database")
    print("\n[OK] " * 30 + "\n")
