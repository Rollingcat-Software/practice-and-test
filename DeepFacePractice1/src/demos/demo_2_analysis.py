"""
Demo 2: Face Analysis
Learn how to analyze facial attributes (age, gender, emotion, race).

Educational Focus:
- Understanding facial attribute detection
- Interpreting confidence scores
- Real-world applications
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.services.face_analysis_service import FaceAnalysisService
from src.utils.visualizer import FaceVisualizer
from src.utils.logger import setup_logger

logger = setup_logger("Demo2_Analysis")
visualizer = FaceVisualizer()


def demo_complete_analysis():
    """
    Perform complete face analysis on an image.

    Learn: How to extract all facial attributes at once.
    """
    print("\n" + "=" * 60)
    print("DEMO 2A: Complete Face Analysis")
    print("=" * 60 + "\n")

    service = FaceAnalysisService()

    img_path = "images/kisi_A_1.jpg"

    logger.info(f"Analyzing: {img_path}")

    try:
        result = service.analyze(img_path)

        print(result)
        print("\n" + "-" * 60)

        # Detailed breakdown
        print("\nDetailed Analysis:")

        if result.age:
            print(f"\n  AGE ESTIMATION:")
            print(f"    Estimated age: {result.age:.1f} years")
            print(f"    Note: AI age estimation has ~5-10 year margin of error")

        if result.gender and result.gender_confidence:
            print(f"\n  GENDER CLASSIFICATION:")
            print(f"    Predicted: {result.gender}")
            print(f"    Confidence: {result.gender_confidence:.1f}%")

        if result.emotion_scores:
            print(f"\n  EMOTION DETECTION:")
            print(f"    Dominant emotion: {result.dominant_emotion}")
            print(f"    All emotions:")
            sorted_emotions = sorted(
                result.emotion_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            for emotion, score in sorted_emotions:
                bar = "█" * int(score / 5)
                print(f"      {emotion:12s}: {score:5.1f}% {bar}")

        if result.race_scores:
            print(f"\n  ETHNICITY CLASSIFICATION:")
            print(f"    Dominant: {result.dominant_race}")
            print(f"    All categories:")
            sorted_races = sorted(
                result.race_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )
            for race, score in sorted_races:
                bar = "█" * int(score / 5)
                print(f"      {race:18s}: {score:5.1f}% {bar}")

        # Visualize
        visualizer.visualize_analysis(
            result,
            save_path="output/analysis_complete.png",
            show=False
        )

    except Exception as e:
        logger.error(f"Analysis failed: {e}")


def demo_emotion_detection():
    """
    Focus on emotion detection.

    Learn: How to detect and interpret emotions.
    """
    print("\n" + "=" * 60)
    print("DEMO 2B: Emotion Detection Focus")
    print("=" * 60 + "\n")

    service = FaceAnalysisService()

    img_path = "images/kisi_A_1.jpg"

    logger.info("Detecting emotions...")

    try:
        result = service.analyze_emotion_only(img_path)

        print(f"Image: {img_path}")
        print(f"Dominant Emotion: {result.dominant_emotion}\n")

        if result.emotion_scores:
            print("Emotion Breakdown:")
            print(f"{'Emotion':<12} {'Score':<8} {'Bar'}")
            print("-" * 40)

            sorted_emotions = sorted(
                result.emotion_scores.items(),
                key=lambda x: x[1],
                reverse=True
            )

            for emotion, score in sorted_emotions:
                bar = "█" * int(score / 2.5)
                symbol = "👉 " if emotion == result.dominant_emotion else "   "
                print(f"{symbol}{emotion:<12} {score:5.1f}%  {bar}")

        print("\nUse Cases:")
        print("  - Customer sentiment analysis")
        print("  - Mental health monitoring")
        print("  - User experience research")
        print("  - Interactive applications")

    except Exception as e:
        logger.error(f"Emotion detection failed: {e}")


def demo_demographics():
    """
    Focus on demographic analysis (age, gender, race).

    Learn: How to extract demographic information.
    """
    print("\n" + "=" * 60)
    print("DEMO 2C: Demographics Analysis")
    print("=" * 60 + "\n")

    service = FaceAnalysisService()

    img_path = "images/kisi_A_1.jpg"

    logger.info("Analyzing demographics...")

    try:
        result = service.analyze_demographics(img_path)

        print(f"Image: {img_path}\n")

        print("DEMOGRAPHIC PROFILE")
        print("=" * 40)

        if result.age:
            print(f"\nEstimated Age: ~{result.age:.0f} years")

        if result.gender and result.gender_confidence:
            print(f"\nGender:")
            print(f"  Classification: {result.gender}")
            print(f"  Confidence: {result.gender_confidence:.1f}%")

        if result.dominant_race:
            print(f"\nEthnicity:")
            print(f"  Primary: {result.dominant_race}")

        print("\n" + "=" * 40)

        print("\nUse Cases:")
        print("  - Demographic statistics for marketing")
        print("  - Audience analysis for content creators")
        print("  - Accessibility features (age-appropriate content)")
        print("  - Research and analytics")

        print("\nEthical Considerations:")
        print("  [WARN]  Always obtain consent before analyzing faces")
        print("  [WARN]  Be aware of bias in AI models")
        print("  [WARN]  Use responsibly and ethically")
        print("  [WARN]  Comply with privacy regulations (GDPR, etc.)")

    except Exception as e:
        logger.error(f"Demographics analysis failed: {e}")


def demo_batch_analysis():
    """
    Analyze multiple images.

    Learn: How to process multiple faces efficiently.
    """
    print("\n" + "=" * 60)
    print("DEMO 2D: Batch Analysis")
    print("=" * 60 + "\n")

    service = FaceAnalysisService()

    image_paths = [
        "images/kisi_A_1.jpg",
        "images/kisi_A_2.jpg",
        "images/kisi_B_1.jpg",
    ]

    # Filter to existing images
    existing_images = [p for p in image_paths if Path(p).exists()]

    if not existing_images:
        print("No images found for batch analysis")
        return

    logger.info(f"Analyzing {len(existing_images)} images...")

    results = service.batch_analyze(existing_images, actions=["age", "gender", "emotion"])

    print(f"\nBatch Analysis Results:")
    print(f"{'Image':<25} {'Age':<8} {'Gender':<10} {'Emotion':<12}")
    print("-" * 65)

    for img_path, result in results.items():
        img_name = Path(img_path).name
        age_str = f"~{result.age:.0f}" if result.age else "N/A"
        gender_str = result.gender if result.gender else "N/A"
        emotion_str = result.dominant_emotion if result.dominant_emotion else "N/A"

        print(f"{img_name:<25} {age_str:<8} {gender_str:<10} {emotion_str:<12}")

    print(f"\nProcessed {len(results)} images successfully!")


if __name__ == "__main__":
    print("\n" + " " * 30)
    print("FACE ANALYSIS TUTORIAL")
    print("Learn how to extract facial attributes with DeepFace")
    print(" " * 30)

    # Create directories
    Path("output").mkdir(exist_ok=True)
    Path("images").mkdir(exist_ok=True)

    # Check for images
    if not Path("images/kisi_A_1.jpg").exists():
        print("\n[WARN]  WARNING: Sample images not found!")
        print("Please add sample images to the 'images' directory.")
        print("\nContinuing with available images...")

    # Run demos
    demo_complete_analysis()
    demo_emotion_detection()
    demo_demographics()
    demo_batch_analysis()

    print("\n" + "[OK] " * 30)
    print("TUTORIAL COMPLETE!")
    print("Check the 'output' folder for visualizations.")
    print("[OK] " * 30 + "\n")
