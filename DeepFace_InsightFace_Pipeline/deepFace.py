import cv2
from deepface import DeepFace

print("DeepFace is ready to analyze.")

print("\n--- Running Verification Test ---")

try:
    # Test 1: Same person, different photos
    # The DeepFace.verify() function takes two image paths and returns a dictionary.
    # With model_name='ArcFace', we specify which recognition model to use.
    result_same_person = DeepFace.verify(
        img1_path="../../../../PycharmProjects/face_verification/test_images/aysenur_eh.jpg",
        img2_path="../../../../PycharmProjects/face_verification/images/aysenur.jpg",
        model_name="ArcFace"
    )
    print("\nComparing aysenur_eh.jpg with aysenur.jpg:")
    print(result_same_person)

    # Cleaner output:
    if result_same_person["verified"]:
        print(f"Result: Match! Distance = {result_same_person['distance']:.4f}")
    else:
        print(f"Result: No Match! Distance = {result_same_person['distance']:.4f}")

    # Test 2: Different people
    result_different_person = DeepFace.verify(
        img1_path="../../../../PycharmProjects/face_verification/test_images/aysenur_eh.jpg",
        img2_path="../../../../PycharmProjects/face_verification/images/ayse_gulsum.jpg",
        model_name="ArcFace"
    )
    print("\nComparing aysenur_eh.jpg with ayse_gulsum.jpg:")
    print(result_different_person)

    if result_different_person["verified"]:
        print(f"Result: Match! Distance = {result_different_person['distance']:.4f}")
    else:
        print(f"Result: No Match! Distance = {result_different_person['distance']:.4f}")

except Exception as e:
    print(f"An error occurred: {e}")

print("\n--- Generating Embeddings for Identification Test ---")

try:
    embedding_vector = DeepFace.represent(
        img_path="../../../../PycharmProjects/face_verification/images/aysenur.jpg",
        model_name="ArcFace"
    )
    # The returned object is a list containing a dictionary.
    embedding = embedding_vector[0]["embedding"]
    print(f"Embedding for aysenur.jpg created successfully. Vector length: {len(embedding)}")

except Exception as e:
    print(f"Could not generate embedding: {e}")
