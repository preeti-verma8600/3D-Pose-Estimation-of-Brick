import os
import tempfile
from flask import Flask, request, jsonify
import numpy as np
from main import (
    preprocess_image, is_very_dull_image, create_roi, load_images, segment_image,
    process_for_brick, extract_close_edges, extract_middle_top_brick,
    compute_3d_pose, draw_coordinate_system, save_pose_to_json
)
import logging
import cv2

app = Flask(__name__)

# Setup logging
logging.basicConfig(level=logging.INFO)

# Environment Variables
OUTPUT_FOLDER = os.getenv("OUTPUT_FOLDER", "./output")
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

logging.info(f"Output folder is set to: {OUTPUT_FOLDER}")

@app.route('/get_brick_pose', methods=['POST'])
def get_brick_pose():
    """Flask endpoint to get the 3D pose of the brick."""
    data = request.get_json()
    
    # Input Validation
    if 'image_path' not in data or 'depth_image_path' not in data or 'camera_params_path' not in data:
        return jsonify({"error": "Missing required parameters."}), 400
    
    image_path = data['image_path']
    depth_image_path = data['depth_image_path']
    camera_params_path = data['camera_params_path']
    
    # Validate file paths
    if not os.path.exists(image_path):
        return jsonify({"error": f"Image file not found: {image_path}"}), 400
    if not os.path.exists(depth_image_path):
        return jsonify({"error": f"Depth image file not found: {depth_image_path}"}), 400
    if not os.path.exists(camera_params_path):
        return jsonify({"error": f"Camera parameters file not found: {camera_params_path}"}), 400
    
    thresholds_rgb_bricks = (np.array([100, 0, 0]), np.array([255, 150, 150]))
    thresholds_hsv_bricks = (np.array([0, 100, 100]), np.array([10, 255, 255]))

    try:
        color_image, depth_image, camera_params = load_images(image_path, depth_image_path, camera_params_path)
        color_roi, depth_roi = create_roi(color_image, depth_image, camera_params)
        color_roi = preprocess_image(color_roi)
        segmented_rgb, segmented_hsv, segmented_final, combined_mask = segment_image(color_roi, thresholds_rgb_bricks, thresholds_hsv_bricks)
    except Exception as e:
        return jsonify({"error": f"Error processing images: {e}"}), 500

    parent_folder = os.path.basename(os.path.dirname(image_path))
    base_filename = os.path.basename(image_path)
    base_filename = os.path.splitext(base_filename)[0]

    contour_mask = None
    if is_very_dull_image(segmented_final):
        temp_file = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        cv2.imwrite(temp_file.name, cv2.cvtColor(segmented_final, cv2.COLOR_RGB2BGR))
        result_image, mask, binary_image, bounding_box = process_for_brick(temp_file.name, OUTPUT_FOLDER)
        temp_file.close()
        os.remove(temp_file.name)

        if bounding_box is not None:
            close_edges_image, close_edges_contours = extract_close_edges(binary_image, mask)
            if close_edges_contours:
                close_edges_mask = np.zeros_like(mask)
                cv2.drawContours(close_edges_mask, close_edges_contours, -1, (255, 255, 255), thickness=cv2.FILLED)
                final_segmented_image = cv2.bitwise_and(color_roi, color_roi, mask=close_edges_mask)
                
                # Save the final segmented image
                final_segmented_image_path = os.path.join(OUTPUT_FOLDER, f"{base_filename}_final_segmented_image.png")
                cv2.imwrite(final_segmented_image_path, final_segmented_image)
                
                contour_mask = close_edges_mask
    else:
        result_image, contour_mask = extract_middle_top_brick(color_roi, 6, OUTPUT_FOLDER)
        # Save the final segmented image for bright image
        if result_image is not None:
            final_segmented_image_path = os.path.join(OUTPUT_FOLDER, f"{base_filename}_final_segmented_image.png")
            cv2.imwrite(final_segmented_image_path, result_image)

    if result_image is not None and contour_mask is not None:
        # Save Depth ROI with Masked Brick
        depth_roi_masked = cv2.bitwise_and(depth_roi, depth_roi, mask=contour_mask)
        depth_roi_masked_path = os.path.join(OUTPUT_FOLDER, f"{base_filename}_depth_roi_masked.png")
        cv2.imwrite(depth_roi_masked_path, depth_roi_masked)
        
        masked_depth_image_path = tempfile.NamedTemporaryFile(suffix='.png', delete=False).name
        brick_position_mm, roll_deg, pitch_deg, yaw_deg = compute_3d_pose(depth_roi, contour_mask, camera_params, masked_depth_image_path)
        os.remove(masked_depth_image_path)
        pose_data = save_pose_to_json(OUTPUT_FOLDER, parent_folder, base_filename, brick_position_mm, roll_deg, pitch_deg, yaw_deg)

        # Save Final 3D Pose Visualization
        pose_visualization_image =  draw_coordinate_system(color_image, brick_position_mm, roll_deg, pitch_deg, yaw_deg, camera_params)
        final_pose_visualization_path = os.path.join(OUTPUT_FOLDER, f"{base_filename}_pose_visualization.png")
        cv2.imwrite(final_pose_visualization_path, pose_visualization_image)
        
        return jsonify(pose_data)
    else:
        return jsonify({"error": "Failed to process the image."}), 400


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)






