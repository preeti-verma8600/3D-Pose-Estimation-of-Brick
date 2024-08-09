import cv2
import numpy as np
import os
import json
from PIL import Image, ImageDraw
from skimage import measure
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

def preprocess_image(image):
    """Convert image to LAB color space, apply CLAHE, and apply Gaussian Blur.
    
    Args:
        image: BGR image to be processed.
        
    Returns:
        Processed image with enhanced contrast and reduced noise.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    image_clahe = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    image_blur = cv2.GaussianBlur(image_clahe, (5, 5), 0)
    return image_blur

def is_very_dull_image(image):
    """Check if the image is very dull based on average brightness.
    
    Args:
        image: Input BGR image.
        
    Returns:
        Boolean indicating if the image is very dull.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    return avg_brightness < 35

def create_roi(image, depth_image, camera_params):
    """Create Region of Interest (ROI) for the given image and depth image.
    
    Args:
        image: BGR image.
        depth_image: Depth image.
        camera_params: Dictionary containing camera parameters.
    
    Returns:
        Tuple containing the ROI of the color image and depth image.
    """
    width = camera_params['width']
    height = camera_params['height']
    roi_width = int(width * 0.50)
    roi_start_x = (width - roi_width) // 2
    roi_start_y = height // 2
    color_roi = image[roi_start_y:height, roi_start_x:roi_start_x + roi_width]
    depth_roi = depth_image[roi_start_y:height, roi_start_x:roi_start_x + roi_width]
    return color_roi, depth_roi

def load_images(image_path, depth_image_path, camera_params_path):
    """Load the RGB and depth images, and the camera parameters.
    
    Args:
        image_path: Path to the color image.
        depth_image_path: Path to the depth image.
        camera_params_path: Path to the JSON file containing camera parameters.
    
    Returns:
        Tuple containing the color image, depth image, and camera parameters dictionary.
    """
    try:
        color_image = cv2.imread(image_path)
        depth_image = cv2.imread(depth_image_path, cv2.IMREAD_ANYDEPTH)
        with open(camera_params_path, 'r') as f:
            camera_params = json.load(f)
    except Exception as e:
        logging.error(f"Error loading images or camera parameters: {e}")
        raise
    return color_image, depth_image, camera_params

def segment_image(color_roi, thresholds_rgb, thresholds_hsv):
    """Segment the image based on RGB and HSV thresholds.
    
    Args:
        color_roi: The region of interest in the color image.
        thresholds_rgb: Tuple containing the lower and upper RGB thresholds.
        thresholds_hsv: Tuple containing the lower and upper HSV thresholds.
    
    Returns:
        Tuple containing the segmented images and the combined mask.
    """
    color_roi_rgb = cv2.cvtColor(color_roi, cv2.COLOR_BGR2RGB)
    color_roi_hsv = cv2.cvtColor(color_roi, cv2.COLOR_BGR2HSV)
    mask_rgb = cv2.inRange(color_roi_rgb, thresholds_rgb[0], thresholds_rgb[1])
    segmented_rgb = cv2.bitwise_and(color_roi_rgb, color_roi_rgb, mask=mask_rgb)
    mask_hsv = cv2.inRange(color_roi_hsv, thresholds_hsv[0], thresholds_hsv[1])
    segmented_hsv = cv2.bitwise_and(color_roi_hsv, color_roi_hsv, mask=mask_hsv)
    combined_mask = cv2.addWeighted(mask_rgb, 0.7, mask_hsv, 0.3, 0)
    segmented_combined = cv2.bitwise_and(color_roi_rgb, color_roi_rgb, mask=combined_mask)
    return segmented_rgb, segmented_hsv, segmented_combined, combined_mask

def remove_small_noise(binary_image, min_area=750):
    """Remove small noise from the binary image based on minimum area threshold.
    
    Args:
        binary_image: Binary image from which to remove small noise.
        min_area: Minimum area threshold for keeping a contour.
    
    Returns:
        Cleaned binary image with small noise removed.
    """
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.ones(binary_image.shape, dtype="uint8") * 255
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            cv2.drawContours(mask, [contour], -1, 0, -1)
    cleaned = cv2.bitwise_and(binary_image, binary_image, mask=mask)
    return cleaned

def draw_best_middle_contour(image_array, original_image, max_aspect_ratio, min_area, min_height):
    """Draw the best middle contour based on aspect ratio, area, and height constraints.
    
    Args:
        image_array: Binary image array to process.
        original_image: Original image where the contour will be drawn.
        max_aspect_ratio: Maximum aspect ratio to consider for the best contour.
        min_area: Minimum area threshold for contours.
        min_height: Minimum height threshold for contours.
    
    Returns:
        Tuple containing the original image with contour drawn, mask, and the bounding box of the best contour.
    """
    height, width = image_array.shape[:2]
    new_width = int(width * 0.85)
    top_half_height = height / 2
    new_height = int(top_half_height / 2 + top_half_height / 4)
    left = int((width - new_width) / 2)
    top = 0
    cropped_image_array = image_array[top:top + new_height, left:left + new_width]
    contours = measure.find_contours(cropped_image_array, 0.8)
    best_contour = None
    best_bounding_box = None
    best_score = 0
    middle_left = new_width * 0.25
    middle_right = new_width * 0.75
    for contour in contours:
        minr, minc = np.min(contour, axis=0).astype(int)
        maxr, maxc = np.max(contour, axis=0).astype(int)
        width = maxc - minc
        height = maxr - minr
        aspect_ratio = width / height if height != 0 else float('inf')
        area = width * height
        if aspect_ratio <= max_aspect_ratio and area >= min_area and height >= min_height:
            middle_coverage = max(0, min(maxc, middle_right) - max(minc, middle_left))
            score = middle_coverage * aspect_ratio
            if score > best_score:
                best_contour = contour
                best_bounding_box = (minc, minr, maxc, maxr)
                best_score = score
    mask = np.zeros_like(image_array, dtype=np.uint8)
    if best_bounding_box:
        minc, minr, maxc, maxr = best_bounding_box
        minc += left
        maxc += left
        draw = ImageDraw.Draw(original_image)
        draw.rectangle([(minc, minr), (maxc, maxr)], outline="red", width=2)
        cv2.rectangle(mask, (minc, minr), (maxc, maxr), (255, 255, 255), thickness=-1)
    return original_image, mask, best_bounding_box

def process_for_brick(image_path, output_path):
    """Process the image to find and draw the best middle contour.
    
    Args:
        image_path: Path to the input image.
        output_path: Path to save the processed image.
    
    Returns:
        Tuple containing the processed binary image, mask, original binary image, and bounding box of the best contour.
    """
    if not os.path.exists(image_path):
        logging.error(f"Image path {image_path} does not exist.")
        return None, None, None, None
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Failed to read image from {image_path}.")
        return None, None, None, None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    cleaned = remove_small_noise(binary)
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
    morphed = Image.fromarray(cleaned)
    image = np.array(morphed.convert('L'))
    max_aspect_ratio = 5.5
    min_area = 1000
    min_height = 50
    clean_binary, mask, bounding_box = draw_best_middle_contour(image, morphed, max_aspect_ratio, min_area, min_height)
    if output_path:
        result_image_path = os.path.join(output_path, "clean_binary.png")
        cv2.imwrite(result_image_path, np.array(clean_binary))
    return clean_binary, mask, binary, bounding_box

def point_to_rect_distance(px, py, rect):
    """Calculate the minimum distance between a point and a rectangle.
    
    Args:
        px: X-coordinate of the point.
        py: Y-coordinate of the point.
        rect: Tuple representing the rectangle (x, y, width, height).
    
    Returns:
        Minimum distance between the point and the rectangle.
    """
    rx, ry, rw, rh = rect
    cx = max(rx, min(px, rx + rw))
    cy = max(ry, min(py, ry + rh))
    return np.sqrt((px - cx) ** 2 + (py - cy) ** 2)

def extract_close_edges(binary_image, mask, distance_threshold=0.1):
    """Extract edges that are close to the masked region.
    
    Args:
        binary_image: Binary image to extract edges from.
        mask: Mask of the region of interest.
        distance_threshold: Maximum distance to consider an edge close.
    
    Returns:
        Tuple containing the image with close edges and the list of close edges.
    """
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_coords = cv2.findNonZero(mask)
    x, y, w, h = cv2.boundingRect(mask_coords)
    close_edges = []
    for contour in contours:
        for point in contour:
            px, py = point[0]
            distance = point_to_rect_distance(px, py, (x, y, w, h))
            if distance <= distance_threshold:
                close_edges.append(contour)
                break
    close_edges_image = np.zeros_like(binary_image)
    cv2.drawContours(close_edges_image, close_edges, -1, (255, 255, 255), 1)
    return close_edges_image, close_edges

def save_contours(image_shape, contours, output_path):
    """Save the contours to an image file.
    
    Args:
        image_shape: Shape of the original image.
        contours: List of contours to draw.
        output_path: Path to save the contours image.
    """
    contour_image = np.zeros(image_shape, dtype=np.uint8)
    cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 2)
    cv2.imwrite(output_path, contour_image)

def extract_middle_top_brick(color_roi, cluster_count, save_prefix):
    """Extract the middle top brick using clustering and HSV segmentation.
    
    Args:
        color_roi: Region of interest in the color image.
        cluster_count: Number of clusters for k-means.
        save_prefix: Prefix for saving output images.
    
    Returns:
        Tuple containing the final segmented image and the contour mask.
    """
    image_hsv = cv2.cvtColor(color_roi, cv2.COLOR_BGR2HSV)
    hue_hist = cv2.calcHist([image_hsv], [0], None, [180], [0, 180])
    hue_hist = hue_hist.flatten()
    peaks = np.argpartition(hue_hist, -2)[-2:]
    lower_hue = min(peaks)
    upper_hue = max(peaks)
    lower_hsv = np.array([lower_hue - 10, 70, 50])
    upper_hsv = np.array([upper_hue + 10, 255, 255])
    mask_hsv = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
    kernel = np.ones((3, 3), np.uint8)
    mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_CLOSE, kernel)
    mask_hsv = cv2.morphologyEx(mask_hsv, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask_hsv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < 500:
            cv2.drawContours(mask_hsv, [contour], -1, 0, -1)
    pixels = image_hsv.reshape(-1, 3)
    _, labels, centers = cv2.kmeans(pixels.astype(np.float32), cluster_count, None, 
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2), 
                                    10, cv2.KMEANS_RANDOM_CENTERS)
    labels = labels.reshape(image_hsv.shape[:2])
    contours, _ = cv2.findContours(mask_hsv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_mask = np.zeros_like(mask_hsv)
    def distance_to_middle_top(contour, image_shape):
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return float('inf')
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        middle_top = (image_shape[1] // 2, 0)
        return np.sqrt((cx - middle_top[0]) ** 2 + (cy - middle_top[1]) ** 2)
    if contours:
        middle_top_contour = min(contours, key=lambda contour: distance_to_middle_top(contour, mask_hsv.shape))
        cv2.drawContours(contour_mask, [middle_top_contour], -1, 255, thickness=cv2.FILLED)
    final_segmented_image = cv2.bitwise_and(color_roi, color_roi, mask=contour_mask)
    if save_prefix:
        cv2.imwrite(f"{save_prefix}_final_segmented_image.png", final_segmented_image)
        cv2.imwrite(f"{save_prefix}_hsv_mask.png", mask_hsv)
    return final_segmented_image, contour_mask

def compute_3d_pose(depth_roi, mask, camera_params, masked_depth_image_path):
    """Compute the 3D pose of the brick relative to the camera's optical center.
    
    Args:
        depth_roi: Depth region of interest.
        mask: Mask of the segmented brick.
        camera_params: Dictionary containing camera parameters.
        masked_depth_image_path: Path to save the masked depth image.
    
    Returns:
        Tuple containing the brick position in millimeters, roll, pitch, and yaw in degrees.
    """
    masked_depth = cv2.bitwise_and(depth_roi, depth_roi, mask=mask)
    cv2.imwrite(masked_depth_image_path, masked_depth)
    depth_scale = 0.1 / 1000.0
    brick_depth_roi_meters = masked_depth * depth_scale
    fx = camera_params['fx']
    fy = camera_params['fy']
    cx = camera_params['px'] - (camera_params['width'] - depth_roi.shape[1]) // 2
    cy = camera_params['py'] - (camera_params['height'] // 2)
    indices = np.where(mask > 0)
    Z = brick_depth_roi_meters[indices]
    u = indices[1]
    v = indices[0]
    X = (u - cx) * Z / fx
    Y = (v - cy) * Z / fy
    points_3d = np.stack((X, Y, Z), axis=-1)
    valid_points = points_3d[~np.isnan(points_3d).any(axis=1)]
    X_points = valid_points[:, :2]
    Z_points = valid_points[:, 2]
    ransac = make_pipeline(PolynomialFeatures(1), RANSACRegressor())
    ransac.fit(X_points, Z_points)
    plane_coef = ransac.steps[1][1].estimator_.coef_
    plane_intercept = ransac.steps[1][1].estimator_.intercept_
    a, b = plane_coef[1], plane_coef[2]
    c = plane_intercept
    plane_normal = np.array([-a, -b, 1])
    plane_normal /= np.linalg.norm(plane_normal)
    pitch = np.arcsin(-plane_normal[0])
    roll = np.arctan2(plane_normal[1], plane_normal[2])
    yaw = np.arctan2(plane_normal[0], plane_normal[2])
    roll_deg = np.degrees(roll)
    pitch_deg = np.degrees(pitch)
    yaw_deg = np.degrees(yaw)
    brick_position = valid_points.mean(axis=0)
    brick_position_mm = brick_position * 1000
    return brick_position_mm, roll_deg, pitch_deg, yaw_deg

def draw_coordinate_system(image, brick_position_mm, roll_deg, pitch_deg, yaw_deg, camera_params):
    """Draw the coordinate system on the image based on the brick's 3D pose.
    
    Args:
        image: Original BGR image where the coordinate system will be drawn.
        brick_position_mm: Brick's position in millimeters.
        roll_deg: Roll angle in degrees.
        pitch_deg: Pitch angle in degrees.
        yaw_deg: Yaw angle in degrees.
        camera_params: Dictionary containing camera parameters.
    
    Returns:
        Image with the coordinate system drawn.
    """
    # Convert position to pixels
    fx = camera_params['fx']
    fy = camera_params['fy']
    cx = camera_params['px']
    cy = camera_params['py']
    
    brick_x, brick_y, brick_z = brick_position_mm / 1000.0  # Convert to meters
    brick_px = int((brick_x * fx / brick_z) + cx)
    brick_py = int((brick_y * fy / brick_z) + cy)
    
    # Define the axis lengths in mm (adjust as needed)
    axis_length = 50  # 50 mm
    
    roll_rad = np.radians(roll_deg)
    pitch_rad = np.radians(pitch_deg)
    yaw_rad = np.radians(yaw_deg)
    
    # Create rotation matrix from roll, pitch, yaw
    R_roll = np.array([[1, 0, 0],
                       [0, np.cos(roll_rad), -np.sin(roll_rad)],
                       [0, np.sin(roll_rad), np.cos(roll_rad)]])
    
    R_pitch = np.array([[np.cos(pitch_rad), 0, np.sin(pitch_rad)],
                        [0, 1, 0],
                        [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]])
    
    R_yaw = np.array([[np.cos(yaw_rad), -np.sin(yaw_rad), 0],
                      [np.sin(yaw_rad), np.cos(yaw_rad), 0],
                      [0, 0, 1]])
    
    R = R_yaw @ R_pitch @ R_roll
    
    # Ensure the correct alignment of the axes
    x_axis = R @ np.array([axis_length, 0, 0]) + brick_position_mm
    y_axis = R @ np.array([0, axis_length, 0]) + brick_position_mm  # Y-axis goes into the wall
    z_axis = R @ np.array([0, 0, axis_length]) + brick_position_mm  # Z-axis goes up (negative of image Y-axis)
    
    x_px = int((x_axis[0] * fx / x_axis[2]) + cx)
    x_py = int((x_axis[1] * fy / x_axis[2]) + cy)
    y_px = int((y_axis[0] * fx / y_axis[2]) + cx)
    y_py = int((y_axis[1] * fy / y_axis[2]) + cy)
    z_px = int((z_axis[0] * fx / z_axis[2]) + cx)
    z_py = int((z_axis[1] * fy / z_axis[2]) + cy)
    
    # Draw the coordinate system on the image
    cv2.line(image, (brick_px, brick_py), (x_px, x_py), (0, 0, 255), 2)  # X-axis in red
    cv2.line(image, (brick_px, brick_py), (y_px, y_py), (0, 255, 0), 2)  # Y-axis in green
    cv2.line(image, (brick_px, brick_py), (z_px, z_py), (255, 0, 0), 2)  # Z-axis in blue
    
    # Draw axis labels
    cv2.putText(image, 'X', (x_px, x_py), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(image, 'Y', (y_px, y_py), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(image, 'Z', (z_px, z_py), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return image

def save_pose_to_json(output_folder, parent_folder, base_filename, brick_position_mm, roll_deg, pitch_deg, yaw_deg):
    """Save the computed brick pose to a JSON file.
    
    Args:
        output_folder: Folder to save the JSON file.
        parent_folder: Parent folder name (used in the filename).
        base_filename: Base name of the image file (used in the filename).
        brick_position_mm: Brick's position in millimeters.
        roll_deg: Roll angle in degrees.
        pitch_deg: Pitch angle in degrees.
        yaw_deg: Yaw angle in degrees.
    
    Returns:
        Dictionary containing the pose data.
    """
    pose_data = {
        'position_mm': {
            'x': brick_position_mm[0],
            'y': brick_position_mm[1],
            'z': brick_position_mm[2]
        },
        'orientation_deg': {
            'roll': roll_deg,
            'pitch': pitch_deg,
            'yaw': yaw_deg
        }
    }
    json_filename = f"{parent_folder}_{base_filename}_pose.json"
    json_output_path = os.path.join(output_folder, json_filename)
    with open(json_output_path, 'w') as json_file:
        json.dump(pose_data, json_file, indent=4)
    return pose_data
















