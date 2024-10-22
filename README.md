# Brick Pose Estimation Project

## Overview

This project aims to estimate the 3D pose (position and orientation) of a brick relative to the optical center of a camera mounted on the gripper of a Pick-and-Place crane. This estimation is crucial to ensure that bricks are placed correctly within a wall structure. The project processes RGB-D images captured after a brick has been placed and outputs the translation in millimeters and rotation (roll, pitch, and yaw) in degrees of the brick.

## Approach

### Algorithm

1. **Image Preprocessing**:
   - The input color image is converted to the LAB color space, where CLAHE (Contrast Limited Adaptive Histogram Equalization) is applied to enhance contrast.
   - Gaussian blur is applied to reduce noise, facilitating more accurate brick segmentation.

2. **Region of Interest (ROI) Extraction**:
   - The algorithm extracts the region of interest from both the color and depth images, focusing on the area where the brick is likely located.

3. **Image Segmentation**:
   - The brick is segmented from the background using RGB and HSV color thresholds.
   - Small noise is removed from the segmented binary image by filtering out small contours based on area thresholds.

4. **Dull vs. Bright Image Processing**:
   - **Dull Images**: If an image is identified as dull (based on average brightness), the algorithm employs a more detailed processing approach to enhance the image and improve brick detection. This involves additional steps such as adaptive thresholding and morphological operations to enhance the visibility of the brick.
   - **Bright Images**: For bright images, where the brick is already well-lit and distinct, the algorithm directly proceeds with segmentation and contour analysis, skipping the additional enhancement steps used for dull images.

5. **Contour Analysis**:
   - The algorithm identifies and analyzes contours within the segmented image to determine the most likely contour corresponding to the brick. Criteria include aspect ratio, area, and position within the image.
   - The best contour is used to create a mask that isolates the brick.

6. **3D Pose Estimation**:
   - Depth values within the masked region are used to compute the 3D coordinates of the brick’s surface points.
   - A RANSAC-based regression model fits a plane to these points, providing an estimation of the brick’s orientation (roll, pitch, and yaw).
   - The brick's position is calculated as the mean of the valid 3D points.

7. **Coordinate System Visualization**:
   - The estimated 3D pose is visualized by drawing a coordinate system (X, Y, Z axes) on the original image, showing the brick’s orientation in 3D space.

### Trade-offs

- **Noise Reduction vs. Detail Preservation**: Gaussian blur was used to reduce noise, which may also blur some small details. A balance was found by tuning the kernel size to optimize both noise reduction and detail preservation.
- **Threshold Sensitivity**: Segmentation thresholds were manually tuned for the provided dataset. These may require adjustment for different lighting conditions or brick colors.
- **Dull vs. Bright Image Processing**: The algorithm differentiates between dull and bright images to apply the appropriate processing techniques, enhancing robustness across varying lighting conditions.
- **RANSAC for Robustness**: RANSAC was selected for plane fitting due to its ability to handle outliers in depth data, albeit at the cost of increased computational complexity.

## Visualization Results

The visualizations generated by the algorithm are saved in the `Results` folder. For each sample in the dataset, the following images are produced:

- **Final Segmented Image**: Shows the brick isolated from the background after segmentation.
- **Pose Visualization**: Displays the original image with a 3D coordinate system overlaid, illustrating the estimated pose of the brick.
- **Masked Depth ROI**: The depth region of interest after applying the mask for the brick, used for pose estimation.

These visualizations help verify the correctness of the pose estimation and provide insights into the algorithm’s performance.

## Usage Instructions


1. **Install Dependencies**: Ensure all required Python packages are installed by running:

   ```bash
   pip install -r requirements.txt

### Start the Server

Execute the following command to start the Flask server:

```bash
python server.py
```

### Sending a Request

To request the 3D pose of a brick, use curl to send a POST request with a JSON payload. The payload should include the paths to the color image, depth image, and camera parameters.
Example curl Command

```bash
curl -X POST -H "Content-Type: application/json" -d @payload.json http://127.0.0.1:5000/get_brick_pose
```

## Debugging Tips

- **Check File Paths:** Ensure that the paths provided in the `payload.json` file are correct and accessible by the server.
- **Verify Server Status:** Ensure that the server is running and accessible at `http://127.0.0.1:5000`.
- **Inspect Server Logs:** Check the server logs for any error messages or stack traces that can provide more details on what went wrong.
- **Test with Simple Payloads:** Simplify your `payload.json` to include only the required fields and test with known good images and parameters to isolate the problem.

## Acknowledgments

This project was part of a test given by [Monumental](https://www.monumental.co/). The dataset used for developing and testing this solution was provided by Monumental.

