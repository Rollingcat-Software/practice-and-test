import cv2
import numpy as np


# --- Step 1: Image Quality Pre-Check ---

def check_blurriness(image_gray, threshold=100.0):
    """Calculates the blurriness score of a grayscale image."""
    variance = cv2.Laplacian(image_gray, cv2.CV_64F).var()
    if variance < threshold:
        return True, variance  # Image is blurry
    return False, variance  # Image is sharp


def check_brightness_and_glare(
    image_gray,
    dark_threshold=50.0,
    glare_threshold_value=240,
    glare_percentage_threshold=1.0
):
    """Checks brightness and glare issues on a grayscale image."""
    # 1. Darkness Check
    mean_brightness = np.mean(image_gray)
    if mean_brightness < dark_threshold:
        return True, f"Image is too dark (Mean: {mean_brightness:.2f})"

    # 2. Glare Check
    total_pixels = image_gray.size
    glare_pixels = np.sum(image_gray > glare_threshold_value)
    glare_ratio = (glare_pixels / total_pixels) * 100.0

    if glare_ratio > glare_percentage_threshold:
        return True, f"Image has excessive glare (Glare: {glare_ratio:.2f}%)"

    return False, "Image brightness is acceptable."


def run_step_1_quality_check(image):
    """
    Runs Step 1 of the pipeline: Blurriness and Brightness/Glare Check.
    Takes an OpenCV image object directly instead of a file path.
    """
    print("--- Step 1: Starting Quality Check ---")
    if image is None:
        return False, "Image could not be read.", None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 1. Blurriness Check
    is_blurry, blur_score = check_blurriness(gray, threshold=100.0)
    if is_blurry:
        message = f"STATUS: REJECTED. Image is too blurry. Score: {blur_score:.2f}"
        print(message)
        return False, "Please capture a sharper photo.", None

    print(f" -> Blurriness Check: PASSED (Score: {blur_score:.2f})")

    # 2. Brightness and Glare Check
    is_bad_brightness, brightness_message = check_brightness_and_glare(gray)

    if is_bad_brightness:
        print(f"STATUS: REJECTED. {brightness_message}")
        return False, "Please capture the image in a well-lit environment without glare.", None

    print(" -> Brightness/Glare Check: PASSED")
    print("--- Step 1: Completed ---")

    return True, "Image quality is acceptable.", image


# --- Step 2: Document Detection and Perspective Correction ---

def order_points(pts):
    """
    Orders the four corners of a quadrilateral as
    (top-left, top-right, bottom-right, bottom-left).
    This is critical for perspective correction.
    """
    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left (smallest x+y)
    rect[2] = pts[np.argmax(s)]  # Bottom-right (largest x+y)

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right (smallest y-x)
    rect[3] = pts[np.argmax(diff)]  # Bottom-left (largest y-x)

    return rect


def run_step_2_perspective_correction(image):
    """
    Runs Step 2 of the pipeline:
    Finds the largest quadrilateral in the image (the ID card)
    and warps it into a flat, top-down view.
    """
    print("--- Step 2: Starting Perspective Correction ---")

    # For performance we could resize the image, but keep aspect ratio.
    # For now, we work with the original size.
    orig_image = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Light blur to reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edged = cv2.Canny(gray, 75, 200)

    # Find contours (shapes) from edges
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours by area (descending) and keep the largest 10
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    screen_cnt = None  # For the 4 corners of the ID

    for c in contours:
        # Approximate contour to a simpler shape
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # If the approximated shape has 4 points, assume it's the ID card
        if len(approx) == 4:
            screen_cnt = approx
            break

    if screen_cnt is None:
        message = "STATUS: REJECTED. Could not detect the four corners of the ID in the image."
        print(message)
        return False, "Please capture your ID so that all four corners are clearly visible.", None

    print(" -> ID contour found.")

    # Order the corners (top-left, top-right, bottom-right, bottom-left)
    rect = order_points(screen_cnt.reshape(4, 2))
    (tl, tr, br, bl) = rect

    # Compute the width of the warped image
    width_a = np.linalg.norm(br - bl)
    width_b = np.linalg.norm(tr - tl)
    max_width = max(int(width_a), int(width_b))

    # Compute the height of the warped image
    height_a = np.linalg.norm(tr - br)
    height_b = np.linalg.norm(tl - bl)
    max_height = max(int(height_a), int(height_b))

    # Destination corner points for the warped image
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    # Get the perspective transform matrix
    matrix = cv2.getPerspectiveTransform(rect, dst)

    # Apply the transformation
    warped_image = cv2.warpPerspective(orig_image, matrix, (max_width, max_height))

    print("--- Step 2: Completed ---")
    return True, "Perspective corrected.", warped_image


# --- Step 3: ID Type Classification ---

def run_step_3_classify_id(corrected_image):
    """
    Runs Step 3 of the pipeline: Determines the type of the flattened ID.
    This is a simple example using Template Matching.
    In a real scenario, a more robust ML model would be preferred.
    """
    print("--- Step 3: Starting ID Type Classification ---")

    # Load example templates (you must create these yourself)
    # Example: Header text on the new Turkish ID card
    try:
        template_tc = cv2.imread("../../../../PycharmProjects/face_verification/test_images/aysenur_tc.jpg", 0)
        # Example: "TR" logo or "SÜRÜCÜ BELGESİ" text on the driver’s license
        template_ehliyet = cv2.imread("../../../../PycharmProjects/face_verification/test_images/aysenur_eh.jpg", 0)

        if template_tc is None or template_ehliyet is None:
            raise FileNotFoundError

    except FileNotFoundError:
        print(" -> Warning: Template files not found. Defaulting to 'TC_KIMLIK'.")
        # For this demo, return a default value without templates
        return True, "Classification skipped.", "TC_KIMLIK"

    gray_corrected = cv2.cvtColor(corrected_image, cv2.COLOR_BGR2GRAY)

    # Get match scores for each template
    res_tc = cv2.matchTemplate(gray_corrected, template_tc, cv2.TM_CCOEFF_NORMED)
    _, max_val_tc, _, _ = cv2.minMaxLoc(res_tc)

    res_ehliyet = cv2.matchTemplate(gray_corrected, template_ehliyet, cv2.TM_CCOEFF_NORMED)
    _, max_val_ehliyet, _, _ = cv2.minMaxLoc(res_ehliyet)

    print(f" -> Match Scores: (TC: {max_val_tc:.2f}, Driver's License: {max_val_ehliyet:.2f})")

    # Simple winner-takes-all decision
    # Thresholds (e.g., 0.7) can be added to handle "no match" cases.
    if max_val_tc > max_val_ehliyet and max_val_tc > 0.6:
        id_type = "TC_KIMLIK"
    elif max_val_ehliyet > max_val_tc and max_val_ehliyet > 0.6:
        id_type = "EHLIYET"
    else:
        id_type = "BILINMEYEN"
        message = "STATUS: REJECTED. Could not clearly determine whether the ID is TC or Driver's License."
        print(message)
        return False, message, None

    print(f" -> Detected ID Type: {id_type}")
    print("--- Step 3: Completed ---")
    return True, "ID type determined.", id_type


# --- Step 4: Targeted Face Detection (ROI - Region of Interest) ---

# For this step we use OpenCV's built-in face detection model.
# You must have 'haarcascade_frontalface_default.xml' available either
# from your OpenCV installation or downloaded into the same directory.
try:
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        raise IOError("Haar cascade file could not be loaded.")
except Exception as e:
    print(
        f"ERROR: Face detection model could not be loaded. Make sure "
        f"'haarcascade_frontalface_default.xml' exists. Error: {e}"
    )
    face_cascade = None


def get_roi_for_id(image_shape, id_type):
    """
    Returns the coordinates of the region of interest (ROI) where the face
    is expected to be, according to the ID type.
    Coordinates are returned as (y_start, y_end, x_start, x_end).
    Percentages are computed relative to image (height, width).
    """
    h, w = image_shape[:2]

    if id_type == "TC_KIMLIK":
        # On the new Turkish ID card, the photo is on the top-left.
        # For example: Width: 0–45%, Height: 5–50%.
        # These values should be tuned with trial and error.
        return int(h * 0.05), int(h * 0.50), int(w * 0.0), int(w * 0.45)

    elif id_type == "EHLIYET":
        # On the new driver’s license, the photo is also on the top-left,
        # but typically in a slightly smaller area.
        return int(h * 0.10), int(h * 0.55), int(w * 0.05), int(w * 0.40)

    else:
        # Unknown type: search the entire image (risky)
        return 0, h, 0, w


def run_step_4_detect_face_in_roi(corrected_image, id_type):
    """
    Runs Step 4 of the pipeline: searches for a face only within the defined ROI.
    """
    print("--- Step 4: Starting Targeted Face Detection ---")
    if face_cascade is None:
        return False, "Face detection model could not be loaded.", None

    # 1. Get ROI coordinates for the given ID type
    y1, y2, x1, x2 = get_roi_for_id(corrected_image.shape, id_type)

    # 2. Crop ROI from the image
    roi_image = corrected_image[y1:y2, x1:x2]

    # 3. Run face detection ONLY on this smaller ROI
    gray_roi = cv2.cvtColor(roi_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray_roi,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    if len(faces) == 0:
        message = f"STATUS: REJECTED. Passport-style photo could not be found on the ID (ROI: {id_type})."
        print(message)
        # For debugging, you can save ROI: cv2.imwrite("debug_roi_fail.jpg", roi_image)
        return False, "No clear face could be detected on the ID.", None

    # If multiple faces are found, keep the largest one
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    (fx, fy, fw, fh) = faces[0]

    # 4. Convert face coordinates from ROI coordinates back to full image coordinates
    original_face_coords = (fx + x1, fy + y1, fw, fh)

    print(f" -> Face found. Coordinates: {original_face_coords}")
    print("--- Step 4: Completed ---")
    return True, "Face successfully detected.", original_face_coords


# --- Step 5: Cropping the Face and Final Check ---

def run_step_5_crop_and_finalize(corrected_image, face_coords, min_size=100, padding_percent=0.1):
    """
    Runs Step 5 of the pipeline:
    Crops the detected face, adds some padding, and runs final size/quality checks.
    """
    print("--- Step 5: Starting Face Cropping and Final Check ---")

    (x, y, w, h) = face_coords

    # 1. Add some padding around the crop (optional but recommended)
    pad_w = int(w * padding_percent)
    pad_h = int(h * padding_percent)

    # Ensure we do not go outside the image boundaries
    h_img, w_img = corrected_image.shape[:2]
    y1 = max(0, y - pad_h)
    y2 = min(h_img, y + h + pad_h)
    x1 = max(0, x - pad_w)
    x2 = min(w_img, x + w + pad_w)

    # 2. Crop the face
    cropped_face = corrected_image[y1:y2, x1:x2]

    # 3. Final Size Check
    # Is the cropped face large enough (in pixels) for embedding?
    (face_h, face_w) = cropped_face.shape[:2]
    if face_h < min_size or face_w < min_size:
        message = (
            f"STATUS: REJECTED. The detected face is too small "
            f"({face_w}x{face_h}). Minimum: {min_size}x{min_size}"
        )
        print(message)
        return False, "The face photo resolution is too low. Please capture a clearer image.", None

    print(f" -> Cropped face size: {face_w}x{face_h}. Size check PASSED.")

    # 4. Final Blur Check (optional but powerful)
    # Is the cropped ID face itself blurry?
    gray_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2GRAY)
    is_blurry, blur_score = check_blurriness(gray_face, threshold=50.0)  # Lower threshold is acceptable for faces

    if is_blurry:
        message = f"STATUS: REJECTED. The face photo on the ID is blurry. Score: {blur_score:.2f}"
        print(message)
        return False, "The face photo on your ID appears blurry.", None

    print(f" -> ID face sharpness check: PASSED (Score: {blur_score:.2f})")
    print("--- Step 5: Completed. Operation Successful! ---")

    return True, "Face successfully cropped and validated.", cropped_face


# --- Main Pipeline Orchestrator ---

def run_full_id_pipeline(image_path):
    """
    Main function that runs all 5 steps of the pipeline in sequence.
    """
    print(f"\n===== STARTING NEW PROCESS: {image_path} =====")

    # Read image from disk
    original_image = cv2.imread(image_path)
    if original_image is None:
        print("ERROR: Image file could not be read or found.")
        return

    # STEP 1: Quality Check
    success, message, image = run_step_1_quality_check(original_image)
    if not success:
        print(f"PIPELINE FAILED. Reason: {message}")
        return

    # STEP 2: Perspective Correction
    success, message, corrected_image = run_step_2_perspective_correction(image)
    if not success:
        print(f"PIPELINE FAILED. Reason: {message}")
        return

    # Save successful correction (for debugging)
    cv2.imwrite("../../../../PycharmProjects/face_verification/debug_1_corrected.jpg", corrected_image)

    # STEP 3: ID Classification
    # Note: For this step to work properly, you should create template files
    # such as "templates/template_tc.jpg", etc.
    success, message, id_type = run_step_3_classify_id(corrected_image)
    if not success:
        print(f"PIPELINE FAILED. Reason: {message}")
        return

    # STEP 4: Targeted Face Detection
    success, message, face_coords = run_step_4_detect_face_in_roi(corrected_image, id_type)
    if not success:
        print(f"PIPELINE FAILED. Reason: {message}")
        return

    # STEP 5: Face Cropping and Final Check
    success, message, final_face_image = run_step_5_crop_and_finalize(corrected_image, face_coords)
    if not success:
        print(f"PIPELINE FAILED. Reason: {message}")
        return

    # SUCCESS!
    print(f"\nPIPELINE COMPLETED SUCCESSFULLY. {message}")
    print("Saving result as 'final_face.jpg'.")
    cv2.imwrite("final_face.jpg", final_face_image)

    # This 'final_face_image' object is now ready to be passed to DeepFace.represent()
    # or InsightFace's get_embedding() function.


# --- TEST ---
# Please replace these paths with your own test images.
# Make sure 'haarcascade_frontalface_default.xml' and
# the 'templates' folder (for Step 3) are available.

# Run the pipeline
run_full_id_pipeline("../../../../PycharmProjects/face_verification/test_images/aysenur_eh.jpg")

# Other tests:
# run_full_id_pipeline("test_images/blurry_id.jpg")      # Should fail at Step 1
# run_full_id_pipeline("test_images/glare_id.jpg")       # Should fail at Step 1
# run_full_id_pipeline("test_images/angled_id.jpg")      # Should succeed at Step 2
# run_full_id_pipeline("test_images/no_corners_id.jpg")  # Should fail at Step 2
