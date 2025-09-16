import numpy as np
import cv2
import os
import json
import pickle
"""
### Script Description
This script processes frames for offsite processing. Once frames are processed, they can be run through TRiVIA for PIV processing.
Place your video and config file in the same directory as this script then run the script.

If you want a different frame rate change the frameInterval variable in the config to the desired value. 
NOTE: The video has a max framerate of 30 frames a second.

#### Key Functionalities:
1. **Frame Extraction**:
   - Extracts frames from a video at a specified framerate.
   
2. **Undistortion and Homography Transformation**:
   - Undistorts frames using camera calibration data.
   - Applies a perspective transformation (homography) to extract a trapezoidal region.

3. **Saving Processed Frames**:
   - Saves the processed frames into a folder named `homographyFrames`.
   
 Credits: 
	-Funding for this project is provided through the USGS Next Generation Water Observing System (NGWOS) Research and Development program.
	-Engineered by: Deep Analytics LLC http://www.deepvt.com/
	-Authors: Makayla Hayes
"""

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_PATH = None
CONFIG_PATH = None
CAMERA_MATRIX = os.path.join(CURRENT_DIR, 'cameraMatrix.pkl')
DIST_COEF = os.path.join(CURRENT_DIR, 'dist.pkl')


def extract_frames(video_path, rate_in_seconds):
    """
    Extract frames from a video at a specified rate and keep them in memory.

    Args:
        video_path (str): Path to the video file.
        rate_in_seconds (int): Interval in seconds at which frames should be extracted.

    Returns:
        list: A list of frames (each frame is a numpy array).
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video file")
        return []

    # Get the frame rate (frames per second) and total frames
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video FPS: {fps}, Total Frames: {total_frames}")

    # Calculate the frame interval
    frame_interval = int(
        fps * rate_in_seconds)  # Frames to skip based on the specified rate

    frames = []  # List to store frames in memory
    frame_count = 0

    while frame_count < total_frames:
        # Set the video position to the next frame to extract
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)

        # Read the frame
        ret, frame = cap.read()
        if not ret:
            break

        # Append the frame to the list
        frames.append(frame)
        print(f"Frame {frame_count} extracted and stored in memory.")

        # Move to the next frame based on the interval
        frame_count += frame_interval

    cap.release()
    print(f"Extraction completed. {len(frames)} frames stored in memory.")
    return frames


def process_frame(frame):
    """
    Applies a perspective transformation (homography) to the input frame to extract a trapezoidal region.
    
    Args:
        frame (numpy.ndarray): The input image/frame to be transformed.
    
    Returns:
        numpy.ndarray: The transformed (trapezoidal) image.
    
    Raises:
        Exception: If there is an error during the perspective transformation.
    """
    global Transform_matrix, x_dist, y_dist, mapx, mapy
    try:

        undistorted_img = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
        transformed = cv2.warpPerspective(undistorted_img, Transform_matrix,
                                          (int(x_dist), int(y_dist)))
        return transformed

    except Exception as e:
        print(f'Error in extract_trapezoid: {e}')
        return frame


#Get files paths for video and config
for root, _, files in os.walk(CURRENT_DIR):
    for file in files:
        file_path = os.path.join(root, file)

        if file.lower().endswith('.mp4'):
            VIDEO_PATH = file_path
        elif file.lower().endswith('.json'):
            CONFIG_PATH = file_path

#load in config
with open(CONFIG_PATH, 'r') as con:
    config = json.load(con)

#load in camera matrix and discoeff
with open(CAMERA_MATRIX, 'rb') as f:
    camera_matrix = pickle.load(f)
with open(DIST_COEF, 'rb') as f:
    distortion_coefficients = pickle.load(f)

height = int(config["reduced_image_height"])
width = int(config["reduced_image_width"])
#get frame interval for extracting images
#get trapezoid points
trapezoid_points = config.get('trapezoid_points', [])
desired_framerate = float(config.get('frameInterval', '0'))

newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix,
                                                  distortion_coefficients,
                                                  (width, height), 0,
                                                  (width, height))
mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, distortion_coefficients,
                                         None, newcameramtx, (width, height), 5)

# Prepare homography transformation matrix
pts1 = np.float32([
    trapezoid_points[1], trapezoid_points[0], trapezoid_points[2],
    trapezoid_points[3]
])
x_dist = abs(trapezoid_points[2][0] - trapezoid_points[3][0])
y_dist = abs(trapezoid_points[3][1] - trapezoid_points[1][1])
pts2 = np.float32([[0, 0], [x_dist, 0], [0, y_dist], [x_dist, y_dist]])

Transform_matrix, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC)
print(f'Extracting frames from {VIDEO_PATH} ad framerate {desired_framerate}')
frames = extract_frames(VIDEO_PATH, desired_framerate)
print(f'{len(frames)} frames extracted')

print(f'Undistorting, homographying, and saving frames')
os.makedirs(os.path.join(CURRENT_DIR, 'homographyFrames'), exist_ok=True)
for index, frame in enumerate(frames):
    processed_frame = process_frame(frame)
    padded_number = str(index).zfill(3)
    frame_path = os.path.join(CURRENT_DIR, 'homographyFrames',
                              f'processedFrame_{padded_number}.jpg')
    cv2.imwrite(frame_path, processed_frame)
    print(f'Processed and saved: {frame_path} ')

print('All frames processed!')
