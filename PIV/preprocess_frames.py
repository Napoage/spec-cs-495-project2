import os
import cv2
import numpy as np
import threading
import queue
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import json
import pickle
import multiprocessing
import shutil
"""
Video Frame Processing, Stabilization, and Homography-based Image Transformation

Description:
This script handles video capture, frame processing, stabilization, and homography-based 
image transformation in a multi-threaded environment. It performs the following key operations: 
capturing video frames, undistorting and processing images, stabilizing frames, applying 
homographic transformations to extract trapezoidal regions, and saving the processed frames 
and video output. The script uses threading to process frames concurrently for efficiency.

Functions:
- load_globals: Loads global configuration, camera calibration data, and necessary parameters.
- EstStabilizationTform: Estimates the affine transformation matrix to stabilize images.
- stabilize_single_image: Stabilizes a single image relative to a reference image using feature matching.
- extract_trapezoid: Applies a perspective transformation to extract a trapezoidal region from an image.
- undistort_homograph_clahe: Undistorts an image, applies a trapezoidal transformation, and enhances contrast with CLAHE.
- process_images: Processes a queue of frames and saves the processed images to disk.
- ImageHandler: Manages video capture, multithreaded frame processing, and handling of video output.

Note:
- The script is designed for processing frames captured from a video stream, applying transformations, 
  stabilizing, and saving processed frames.
- Multi-threading is used to handle image capture and processing concurrently.
- Error handling is incorporated for loading configurations and processing frames.
- The final processed video is saved in the specified directory.

 Credits: 
	-Funding for this project is provided through the USGS Next Generation Water Observing System (NGWOS) Research and Development program.
	-Engineered by: Deep Analytics LLC http://www.deepvt.com/
	-Ported from:
		 Legleiter, C.J., 2024, TRiVIA - Toolbox for River Velocimetry using Images from Aircraft (ver. 2.1.3, September, 2024): 
				 U.S. Geological software release, https://doi.org/10.5066/P9AD3VT3.
	-Authors: Makayla Hayes
"""

# --- Camera Calibration and Transformation Parameters ---
CAMERA_MATRIX = None  # Camera matrix for undistortion
DIST_COEFF = None  # Distortion coefficients for undistortion
Transform_matrix = None  # Homography transformation matrix
Transform_map = None  # Map for homography-based transformation
psf = None  # Point Spread Function (if applicable)

# --- Image Processing and Masking ---
trapezoid_points = None  # Points for trapezoidal transformation
mask = None  # Mask for image processing (if applicable)
CLAHE = None  # CLAHE object for contrast enhancement

# --- Image Distortion and Undistortion Maps ---
MAPX = []  # X coordinate map for undistortion
MAPY = []  # Y coordinate map for undistortion
x_dist = None  # Horizontal distance for transformation
y_dist = None  # Vertical distance for transformation

# --- Video Capture Settings ---
device = 3  # Camera device index
FRAMERATE = None  # Frame rate of video capture
duration = None  # Duration of video capture in seconds

# --- Processing and Saving Configuration ---
config = None  # Global configuration (from JSON file)
save = False  # Flag indicating whether to save images or not
save_path = None  # Path to save images if enabled
timestamp = None  # Timestamp for saving processed images

# --- Stabilization Settings ---
stabilize = None  # Stabilization configuration (if enabled)

# Set BASE_DIR to the main_repo directory
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
SUPER_PATH = os.path.join(BASE_DIR, 'images')

if not os.path.exists(SUPER_PATH):
    os.makedirs(SUPER_PATH)


def load_globals():
    """
    Loads the global configuration data and parameters needed for video processing, including:
    - Camera calibration matrices (CAMERA_MATRIX and DIST_COEFF)
    - Trapezoid points for perspective transformation
    - Mask for processing, if applicable
    - Homography transformation matrix (Transform_matrix)
    - Frame rate and capture duration
    - Paths for saving processed images, if enabled
    
    Raises:
        Exception: If there is an error loading the configuration or the necessary files.
    """
    global config, CAMERA_MATRIX, DIST_COEFF, trapezoid_points, mask, stabilize
    global Transform_matrix, MAPX, MAPY, CLAHE, x_dist, y_dist, FRAMERATE
    global device, duration, save, save_path, timestamp

    save_json_path = os.path.join(BASE_DIR, 'save.json')

    try:
        # Load save configuration
        with open(save_json_path, 'r') as f:
            save_config = json.load(f)

        save_config_directory = save_config.get('config_folder')
        save_config_path = os.path.join(save_config_directory, 'config.json')

        # Load main configuration
        with open(save_config_path, 'r') as f:
            config = json.load(f)

        # Load camera calibration matrices
        camera_matrix_path = os.path.join(BASE_DIR, 'Image_Processing',
                                          'cameraMatrix.pkl')
        dist_coef_path = os.path.join(BASE_DIR, 'Image_Processing',
                                      'dist.pkl')

        with open(camera_matrix_path, 'rb') as f:
            CAMERA_MATRIX = pickle.load(f)
        with open(dist_coef_path, 'rb') as f:
            DIST_COEFF = pickle.load(f)

        # Load trapezoid points
        trapezoid_points = config.get('trapezoid_points', [])

        # Load mask if specified
        mask = None
        if config.get('mask', '') == 'yes':
            mask_path = os.path.join(BASE_DIR, 'app',
                                     config.get('mask_path', ''))
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # Stabilization setting
        stabilize = config.get('stabilize', '')

        timestamp = time.strftime("%Y-%m-%d-%H_%M_%S")
        save_path = os.path.join(save_config_directory, timestamp)
        save_config['current_data_directory'] = save_path
        os.makedirs(save_path)
        
        # Save images if enabled
        save = config.get('save_images', '') == 'yes'
        if save:
            print(save_path)
            # Create directories for saving images
            os.makedirs(os.path.join(save_path, f'{timestamp}_original_images'),
                        exist_ok=True)

        # Prepare homography transformation matrix
        pts1 = np.float32([
            trapezoid_points[1], trapezoid_points[0], trapezoid_points[2],
            trapezoid_points[3]
        ])
        x_dist = abs(trapezoid_points[2][0] - trapezoid_points[3][0])
        y_dist = abs(trapezoid_points[3][1] - trapezoid_points[1][1])
        pts2 = np.float32([[0, 0], [x_dist, 0], [0, y_dist], [x_dist, y_dist]])

        Transform_matrix, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC)

        # Camera undistortion map
        height = int(config["reduced_image_height"])
        width = int(config["reduced_image_width"])
        newcameramtx, _ = cv2.getOptimalNewCameraMatrix(CAMERA_MATRIX,
                                                        DIST_COEFF,
                                                        (width, height), 0,
                                                        (width, height))
        MAPX, MAPY = cv2.initUndistortRectifyMap(CAMERA_MATRIX, DIST_COEFF,
                                                 None, newcameramtx,
                                                 (width, height), 5)

        # CLAHE for contrast enhancement
        CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # Frame rate and duration
        FRAMERATE = float(config.get('frameInterval', '0'))
        duration = float(config.get('capture_time', '0'))

        # Save updated configuration
        with open(save_json_path, 'w') as f:
            json.dump(save_config, f, indent=4)

    except Exception as e:
        print(f'Error loading global configuration and data: {e}')


def EstStabilizationTform(imgRef, imgB, ptThresh):
    # Use ORB instead of FAST + BRIEF
    orb = cv2.ORB_create(edgeThreshold=int(ptThresh * 255))
    keypointsA, featuresA = orb.detectAndCompute(imgRef, None)
    keypointsB, featuresB = orb.detectAndCompute(imgB, None)

    # Match features using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(featuresA, featuresB)

    pointsA = np.float32([keypointsA[m.queryIdx].pt for m in matches])
    pointsB = np.float32([keypointsB[m.trainIdx].pt for m in matches])

    tform, _ = cv2.estimateAffinePartial2D(pointsB, pointsA)
    #H = np.vstack([tform, [0, 0, 1]])

    return tform


def stabilize_single_image(imgRef, imgB, ptThresh=0.2):
    """
    Stabilize a single image relative to a reference image.
    
    Parameters:
        imgRef (ndarray): The reference image (grayscale).
        imgB (ndarray): The image to be stabilized (grayscale).
        ptThresh (float): Threshold for feature detection.
    
    Returns:
        stabilized_img (ndarray): The stabilized version of imgB.
        tform (ndarray): The affine transformation matrix.
    """
    # Estimate the transformation
    tform = EstStabilizationTform(imgRef, imgB, ptThresh)

    # Apply the transformation to stabilize the image
    stabilized_img = cv2.warpAffine(imgB, tform,
                                    (imgRef.shape[1], imgRef.shape[0]))

    return stabilized_img


def extract_trapezoid(frame):
    """
    Applies a perspective transformation (homography) to the input frame to extract a trapezoidal region.
    
    Args:
        frame (numpy.ndarray): The input image/frame to be transformed.
    
    Returns:
        numpy.ndarray: The transformed (trapezoidal) image.
    
    Raises:
        Exception: If there is an error during the perspective transformation.
    """
    global Transform_matrix, x_dist, y_dist

    try:
        transformed = cv2.warpPerspective(frame, Transform_matrix,
                                          (int(x_dist), int(y_dist)))
        return transformed
    except Exception as e:
        print(f'Error in extract_trapezoid: {e}')
        return None


def undistort_homograph_clahe(img, index):
    """
    Undistorts an image using the camera calibration data, applies a trapezoidal transformation,
    and applies CLAHE (Contrast Limited Adaptive Histogram Equalization).
    
    Args:
        img (numpy.ndarray): The input distorted image to be processed.
        index (int): The index of the frame being processed, used for naming saved images.
    
    Returns:
        numpy.ndarray: The processed and undistorted image.
    
    Raises:
        Exception: If there is an error during undistortion or image processing.
    """
    global mask, MAPX, MAPY, save_path
    try:

        if save:
            #save raw images if user wants
            raw_image_path = os.path.join(save_path,
                                          f'{timestamp}_original_images',
                                          f'frame_{index}.jpg')
            cv2.imwrite(raw_image_path, img)

        #Undistort the image
        undistorted_img = cv2.remap(img, MAPX, MAPY, cv2.INTER_LINEAR)

        #Transform image by extracting homography trapezoid
        img = extract_trapezoid(undistorted_img)

        if index == 0:
            #If this is the first image save the img for results overlay
            capture_image_path = os.path.join(save_path, 'capture_image.jpg')
            cv2.imwrite(capture_image_path, img)

        if mask is not None:
            # Mask image
            img = cv2.bitwise_and(img, img, mask=mask)

        # Filter image
        img = CLAHE.apply(img)

        return img  #return processed image

    except Exception as e:
        print(f'Error in undistort_homograph_CLAHE: {e}')


def process_images(frame_queue, stop_event):
    """
    Processes images from the frame queue in a multithreaded manner using the provided processor.
    The processed images are saved to the disk if saving is enabled.
    
    Args:
        processor (ImageProcessor or function): The image processing function or object that performs image processing.
        frame_queue (queue.Queue): The queue containing the frames to be processed.
        stop_event (threading.Event): An event used to signal when the processing should stop.
    
    Raises:
        Exception: If there is an error processing an image.
    """
    global save, save_path
    while not stop_event.is_set() or not frame_queue.empty():
        try:
            if not frame_queue.empty():

                # Get the (index, frame) tuple from the queue
                index, frame = frame_queue.get()

                # Process the frame
                processed_frame = undistort_homograph_clahe(frame, index)

                # Save the processed frame with the index as the filename
                if processed_frame is not None:
                    img_name = f"final_frame_{index}.jpg"
                    img_path_out = os.path.join(SUPER_PATH, img_name)
                    cv2.imwrite(img_path_out, processed_frame)

            else:
                time.sleep(0.1)  # Avoid busy waiting
        except Exception as e:
            print(f"Error processing image: {e}")


class ImageHandler:
    """
    Handles video stream capture and processing using threading for concurrent video capturing and image processing.
    
    Attributes:
        device (str): The video capture device (e.g., '/dev/video0').
        FRAMERATE (int): The frame rate of the video capture.
        duration (int): The duration of video capture in seconds.
        frame_queue (queue.Queue): A queue for storing captured frames before processing.
        stop_event (threading.Event): Event to signal when to stop the video capture and processing.
        capture_finished (threading.Event): Event to signal when the video capture has finished.
        frame_index (int): Counter for the frame index.
    
    Methods:
        start_loading(): Starts a thread to capture frames from the video stream.
        start_processing_threads(processing_threads): Starts multiple threads to process the captured frames.
        _image_loading()(): Captures video frames from the device for the specified duration.
    """

    def __init__(self, device, FRAMERATE, duration, queue_size=1000):
        """
        Initializes a ImageHandler instance.
        
        Args:
            device (str): The video capture device (e.g., '/dev/video0').
            FRAMERATE (int): The frame rate of the video capture.
            duration (int): The duration for which to capture video in seconds.
            queue_size (int): The maximum size of the frame queue (default is 1000).
        """
        self.device = device
        self.FRAMERATE = FRAMERATE
        self.duration = duration
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.stop_event = threading.Event()
        self.capture_finished = threading.Event()
        self.frame_index = 0  # Track the index of the frames added to the queue

    def start_loading(self):
        """
        Starts a thread that captures frames from the video stream.
        """
        loading_thread = threading.Thread(target=self._image_loading())
        loading_thread.start()
        # loading_thread.join()

    def start_processing_threads(self, processing_threads):
        """
        Starts multiple threads to process the captured frames.
        
        Args:
            processing_threads (int): The number of threads to start for processing.
        
        Returns:
            list: A list of thread objects that are processing the images.
        """
        # Start the pool of worker threads for image processing
        threads = []
        for _ in range(processing_threads):
            t = threading.Thread(target=process_images,
                                 args=(
                                     self.frame_queue,
                                     self.stop_event,
                                 ))
            t.start()
            threads.append(t)
        return threads

    def _image_loading(self):
        """
        Captures video frames from the device for the specified duration and adds them to the frame queue.
        """
        frame_files = sorted(os.listdir(os.path.join(BASE_DIR, "raw_frames")))

        for frame_file in frame_files:
            # Construct the full file path
            frame_path = os.path.join(os.path.join(BASE_DIR, "raw_frames"),
                                      frame_file)

            # Read the frame
            frame = cv2.imread(frame_path)

            if frame is None:
                print(f"Error reading frame from {frame_path}")
                continue

            # Convert the frame to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Add the (index, grayscale frame) tuple to the queue if there's space
            if not self.frame_queue.full():
                self.frame_queue.put((self.frame_index, gray_frame))
                self.frame_index += 1  # Increment the frame index
            else:
                print("Queue is full, dropping frame.")

        # Signal that capture is finished
        self.capture_finished.set()


def main():
    """
    The main entry point for the video capture and processing pipeline. 
    It loads the configuration, starts video capture, and processes the frames using multithreading.
    It waits until the video capture and processing are finished and then stops the threads.
    """
    load_globals()
    processing_threads = multiprocessing.cpu_count() - 2

    # Create ImageHandler instance
    video_handler = ImageHandler(device, FRAMERATE, duration)

    # Start video capture and processing
    video_handler.start_loading()
    threads = video_handler.start_processing_threads(processing_threads)

    # Wait for the capture to finish and then check if the queue is empty
    video_handler.capture_finished.wait()

    # Once capture is done, wait for the queue to empty
    while not video_handler.frame_queue.empty():
        time.sleep(0.5)  # Check every half second if the queue is empty

    time.sleep(1)

    # Signal to stop the processing threads
    video_handler.stop_event.set()
    # Join processing threads

    for t in threads:
        t.join()

    #Move video to correct directory
    video_path = os.path.join(f'{BASE_DIR}/save_data', 'video_output.mp4')
    destination = os.path.join(save_path, f'{timestamp}_video.mp4')
    shutil.move(video_path, destination)


if __name__ == "__main__":
    main()
