# File Name: advanced_model_test.py

import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

print("--- Starting Advanced Model Test with InsightFace ---\n")

try:
    # --- Step 1: Load the Model ---
    # The 'buffalo_l' model incorporates training strategies similar to ArcFace,
    # MagFace, and AdaFace. This makes it more robust to challenging conditions
    # such as low-quality images, masks, makeup, and illumination variations.
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    print("AI Engine initialized successfully with the buffalo_l model.\n")

    # --- Step 2: Embedding Extraction Function ---
    # This function extracts a face embedding from an image using the loaded model.
    def get_face_embedding(image_path):
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error: Could not read or find the file {image_path}.")
            return None

        faces = app.get(img)
        if not faces:
            print(f"Warning: No face detected in {image_path}.")
            return None

        # Use the embedding of the most prominent detected face
        return faces[0].normed_embedding

    # --- Step 3: Face Comparison Function (Cosine Similarity) ---
    # Measures how similar two embedding vectors are.
    def verify_faces(embedding1, embedding2, threshold=0.6):
        if embedding1 is None or embedding2 is None:
            return 0.0, False

        similarity = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
        is_match = similarity > threshold
        return similarity, is_match

    # --- Step 4: Run Tests ---
    # Paths of the images to be tested
    img_tc = "test_images/aysenur_tc.jpg"
    img_live = "images/aysenur.jpg"
    img_different = "images/ayse_gulsum.jpg"

    print(f"Images to Compare:\n 1: {img_tc}\n 2: {img_live}\n 3: {img_different}\n")

    # Generate embeddings
    emb_tc = get_face_embedding(img_tc)
    emb_live = get_face_embedding(img_live)
    emb_different = get_face_embedding(img_different)

    # Test 1: Same person, different photos
    print(">>> Test 1: Comparing images of the same person...")
    score_same, match_same = verify_faces(emb_tc, emb_live)
    print(f"Similarity Score: {score_same:.4f}, Match? -> {match_same}\n")

    # Test 2: Different people
    print(">>> Test 2: Comparing images of different people...")
    score_diff, match_diff = verify_faces(emb_tc, emb_different)
    print(f"Similarity Score: {score_diff:.4f}, Match? -> {match_diff}\n")

except Exception as e:
    print(f"An error occurred: {e}")
