import os
import sys
import glob
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import threading
import json
from flask import Flask, render_template, send_file, Response, request, jsonify, redirect, url_for, session, stream_with_context, g, flash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import subprocess
from pathlib import Path
from flask import request, jsonify

# Mock hardware for local testing
try:
    from imusensor.MPU9250 import MPU9250
    import smbus
except ImportError:
    sys.path.append('..')
    from mock_hardware import MockMPU9250, MockSMBus
    # Create a mock module that has the expected structure
    MPU9250 = type('MockModule', (), {'MPU9250': MockMPU9250})()
    smbus = type('MockModule', (), {'SMBus': MockSMBus})()

import time
import pickle
import json
import logging
import psutil
from datetime import datetime, timedelta
import shutil
import time
import re
try:
    import pwd
except ImportError:
    # Mock pwd module for Windows
    pwd = type('MockModule', (), {
        'getpwnam': lambda name: type('MockPwdStruct', (), {'pw_uid': 1000, 'pw_gid': 1000})()
    })()
import subprocess
import sys
"""
Flask Server Application

This script sets up a Flask server with various helper methods and routes. It handles device management, 
video streaming, IMU sensor data processing, and login authentication. The application integrates with 
GStreamer for video capture and uses an IMU sensor for orientation tracking.

 Credits: 
	-Funding for this project is provided through the USGS Next Generation Water Observing System (NGWOS) Research and Development program.
	-Engineered by: Deep Analytics LLC http://www.deepvt.com/
	-Authors: Makayla Hayes, Jaylene Naylor

"""

# Global Lock for Thread Safety
lock = threading.Lock()


def is_device_busy(device_path):
    """
    Check if the device is being used by any process.
    
    Args:
        device_path (str): Path to the device.
    
    Returns:
        bool: True if the device is busy, False otherwise.
    """
    
    for proc in psutil.process_iter(['pid', 'open_files']):
        try:
            for file in proc.info['open_files'] or []:
                if file.path == device_path:
                    print('Checking device')
                    return True
        except (psutil.NoSuchProcess, psutil.AccessDenied,
                psutil.ZombieProcess):
            pass
    return False


def check_device(device_path):
    """
    Wait until the device is free and update the global status.
    
    Args:
        device_path (str): Path to the device.
    
    Returns:
        bool: True when the device becomes available.
    """
    
    global device_busy_status
    while is_device_busy(device_path):
        print(f"{device_path} is busy. Waiting...")
        device_busy_status['status'] = f"{device_path} is busy. Waiting..."
        time.sleep(2)
    device_busy_status['status'] = f"{device_path} is now free. Proceeding..."
    return True


class VideoStreamHandler:
    """
    Handles video streaming using OpenCV.
    """
    global WIDTH, HEIGHT
    def __init__(self):
        # Initialize the video capture from /dev/video0
        # Wait for /dev/video0 to be free
        check_device('/dev/video2')
        self.cap = cv2.VideoCapture(2)
        self.frame = None
        self.running = True

        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

        if not self.cap.isOpened():
            raise RuntimeError("Error: Unable to open video capture")

        self.video_thread = threading.Thread(target=self.update_frame,
                                             daemon=True)
        self.video_thread.start()

    def update_frame(self):
        """
        Continuously update the latest frame from the video capture.
        """
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(1)
                continue
            with lock:
                self.frame = frame

    def get_frame(self):
        """
        Retrieve the latest captured frame.
        """
        with lock:
            return self.frame

    def stop(self):
        """
        Stop the video capture and release the device.
        """
        self.running = False
        self.video_thread.join()
        self.cap.release()


#Set up all global variables
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)

# Flask app
with open(f"credentials.json",'r') as cred:
    creds=json.load(cred)
app = Flask(__name__)
app.secret_key = os.urandom(creds["SECRET_KEY"])
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
users = {creds["USERNAME"]: {'password': creds["PASSWORD"]}}

imu_angle = None
script_process = None
dewarped_frame = None
video_handler = None

# IMU setup
current_points = None
current_angle = None
address = 0x68
bus = smbus.SMBus(1)
imu = MPU9250.MPU9250(bus, address)
imu.begin()
imu.loadCalibDataFromFile(os.path.join(BASE_DIR, "IMU/calib.json"))

SAVE_FOLDER_PATH = os.path.join(BASE_DIR, 'save_data')
# Create the directory if it doesn't exist
if not os.path.exists(SAVE_FOLDER_PATH):
    try:
        os.makedirs(SAVE_FOLDER_PATH)
        logging.info(f"Created directory: {SAVE_FOLDER_PATH}")
    except Exception as e:
        logging.error(f"Error creating directory {SAVE_FOLDER_PATH}: {str(e)}")

SAVE_CONFIG = os.path.join(BASE_DIR, 'save.json')
MAIN_CONFIG = os.path.join(BASE_DIR, 'config.json')
FONT_DIR = os.path.join(BASE_DIR, 'app/static/font/')
BOOTSTRAP_DIR = os.path.join(BASE_DIR, 'app/static/bootstrap/')
SAVED_TEST_FOLDER_PATH = os.path.join(BASE_DIR, 'saved_test')
with open(MAIN_CONFIG, "r") as config_file:
    config_data = json.load(config_file)

SENSOR_WIDTH = float(config_data.get("cmos_sensor_width"))
SENSOR_HEIGHT = float(config_data.get("cmos_sensor_height"))
FOCAL_LENGTH = float(config_data.get("focal_length"))
WIDTH = int(config_data.get("reduced_image_width"))
HEIGHT = int(config_data.get("reduced_image_height"))


# Login credentials (use environment variables or a secure method in production)
USERNAME = creds["USERNAME"]
PASSWORD = creds["PASSWORD"]


class User(UserMixin):

    def __init__(self, id):
        self.id = id


del creds, cred

monitor_file_path = os.path.join(BASE_DIR, "monitor_file.txt")
# sensor_width, sensor_height, focal_length = 5.376, 3.024, 3.21  # Camera parameters
LOG_FILE_PATH = '/var/log/'
STARTUP_IDENTIFIER = "* Running on http://127.0.0.1:5000"

# Load camera matrix and distortion coefficients
with open(os.path.join(BASE_DIR, 'Image_Processing/cameraMatrix.pkl'),
          'rb') as f:
    camera_matrix = pickle.load(f)
with open(os.path.join(BASE_DIR, 'Image_Processing/dist.pkl'), 'rb') as f:
    distortion_coefficients = pickle.load(f)

newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix,
                                                  distortion_coefficients,
                                                  (WIDTH, HEIGHT), 0, (WIDTH, HEIGHT))
mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, distortion_coefficients,
                                         None, newcameramtx, (WIDTH, HEIGHT), 5)

# Flag to track device status
device_busy_status = {'status': 'waiting'}


def is_process_running():
    """
    Check if the process is currently running based on a monitor file.
    
    Returns:
        bool: True if the process is running, False otherwise.
    """
    try:
        with open(monitor_file_path, "r") as file:
            content = file.read().strip()
            g.process_running = (content == 'run')
            return content == "run"
    except FileNotFoundError:
        g.process_running = False
        return False

class User(UserMixin):
    def __init__(self, id):
        self.id = id
        
        
# Function to find the latest .pkl file in a directory
def get_latest_results():
    with open(SAVE_CONFIG, 'r') as sconfig:
        save_info =json.load(sconfig)
    
    if 'latest_pickle' in save_info and save_info['latest_pickle']:
        return save_info['latest_pickle']   

def process_data(file_path):
    """
    Process CSV files in the specified directory and store the data as numpy arrays.

    Args:
        file_path (str): The directory path containing the CSV files.

    Returns:
        dict: A dictionary where the keys are derived from the file names (last part after underscore) and 
              the values are numpy arrays of the data from each corresponding CSV file.
    """
    data = {}
    for file in os.listdir(file_path):
        if file.endswith('.csv'):
            key = os.path.splitext(file)[0].split("_")[-1]
            csv_path = os.path.join(file_path, file)
            data[key] = pd.read_csv(csv_path).to_numpy()
    return data


def homography_angle(v, h, t):
    """
    Calculate the homography angle in radians.

    Args:
        v (float): Vertical field of view angle in radians.
        h (float): Horizontal field of view angle in radians.
        t (float): Tilt angle in radians.

    Returns:
        float: The homography angle, expressed as a fraction of pi (radians).
    """
    phi = np.arctan((np.cos(t) * np.tan(h / 2) * ((1 / (np.cos(t + v / 2))) -
                                                  (1 / (np.cos(t - v / 2))))) /
                    (np.tan(v / 2) * ((1 / (np.cos(t + v / 2))) +
                                      (1 / (np.cos(t - v / 2))))))
    return phi / np.pi


def read_imu(imu):
    """
    Read the IMU sensor and compute the orientation.
    
    Args:
    imu (IMU): The IMU sensor object.
    
    Returns:
    float: The pitch angle in radians, adjusted to a coordinate system based on IMU positioning.
    """
    imu.readSensor()
    imu.computeOrientation()
    ipitch, iroll, iyaw = imu.pitch, imu.roll, imu.yaw
    # Add 90 degrees to pitch as we define nadir = 0
    return (np.radians(ipitch + 90), np.radians(iroll), np.radians(iyaw + 90))


def get_vert_horz_angle(sensor_width, sensor_height, focal_length):
    """
    Calculate vertical and horizontal field of view angles in radians.
    
    Args:
    sensor_width (float): Width of the sensor.
    sensor_height (float): Height of the sensor.
    focal_length (float): Focal length of the lens.
    
    Returns:
    tuple: Vertical and horizontal field of view angles in radians.
    """
    vertical = 2 * np.arctan(sensor_height / (2 * focal_length))
    horizontal = 2 * np.arctan(sensor_width / (2 * focal_length))
    return vertical, horizontal


VERT_ANGLE, HORIZ_ANGLE = get_vert_horz_angle(SENSOR_WIDTH, SENSOR_HEIGHT,
                                              FOCAL_LENGTH)


def clamp(value, min_value, max_value):
    """
    Make's sure that the trapezoid does not extend further than the image size
    Args:
    value (float): calculated values.
    min_value (float): 0.
    max_value (float): maximum value that trapezoid coordinate can be.
    
    Returns:
    float: maximum value of the minimum (between the value and max dimension) and 0.
    """
    return max(min(value, max_value), min_value)


def get_trapezoid(angle):
    global SENSOR_WIDTH, SENSOR_HEIGHT,FOCAL_LENGTH, WIDTH, HEIGHT
    """
    Find the trapezoid shape based on the current points and the angle of camera. The top two points are lined up on the same y value
    and the bottom two points are line up on the same y value. Based on the angle you get the distance in which the bottom coordinate is wider
    than the top coordinate. That value is subtracted from the top left to get the bottom left x coordinate and then added to the top right to get 
    the bottom right coordinate. There are checks in place to make sure no coordinate extends beyond the image size, as well as that the top left 
    and top right coordinates are never switched.
    
    Args:
    angle (float): angle based off the imu and then calculated angle value
    
    sets the global current_points to the new trapezoid points
    """
    global current_points

    frame_width = WIDTH  # Update with actual frame width
    frame_height = HEIGHT  # Update with actual frame height

    top_right, top_left, bottom_left, bottom_right = current_points

    # Ensure bottom points share the same y-coordinate (height)
    bottom_right[1] = bottom_left[1]
    top_right[1] = top_left[1]

    # Calculate the height and adjust x-values of the trapezoid
    height = abs(bottom_left[1] - top_left[1])

    if height <= 0:
        print("Invalid height, cannot proceed with the trapezoid calculation.")
        return

    x_offset = height * np.tan(angle)

    # Update bottom points based on the calculated x-offset
    bottom_left[0] = top_left[0] - x_offset
    bottom_right[0] = top_right[0] + x_offset
    # Check if the bottom points exceed the frame bounds and adjust height until they are in frame
    while bottom_left[0] < 0 or bottom_right[0] > frame_width:
        # Reduce the height to bring bottom points inside the frame
        height -= 10  # Adjust this step to be more or less aggressive
        if height <= 0:

            print("Height adjustment failed, invalid trapezoid.")
            return
        # Update y-coordinates of bottom points to reduce height
        bottom_left[1] = top_left[1] + height
        bottom_right[1] = top_right[1] + height
        x_offset = height * np.tan(angle)

        bottom_left[0] = top_left[0] - x_offset
        bottom_right[0] = top_right[0] + x_offset
        # print(x_offset)
    # Ensure the top-left x-coordinate is less than the top-right
    if top_left[0] > top_right[0]:
        print("Top left exceeds top right, invalid trapezoid.")
        return

    # Update global current_points with the new trapezoid
    current_points = top_right, top_left, bottom_left, bottom_right


def dewarp_frame(frame):
    """
    Apply dewarping to the input frame using the predefined camera matrix and distortion coefficients.
    
    Args:
    frame (array): current frame of video
    
    Returns:
    dewarped_frame (array): returns the dewarped frame
    """
    global mapx, mapy, newcameramtx
    # dewarped_frame = cv2.undistort(frame, camera_matrix, distortion_coefficients, None, newcameramtx)
    dewarped_frame = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
    return dewarped_frame


def find_largest_trapezoid(angle):
    global SENSOR_WIDTH, SENSOR_HEIGHT,FOCAL_LENGTH, WIDTH, HEIGHT
    """
    Find the largest trapezoid that fits within the frame for a given angle.
    
    Args:
    angle: Angle in radians.
    
    Returns:
    np.ndarray: Points of the largest trapezoid.
    """
    frame_height, frame_width = HEIGHT, WIDTH
    max_height = frame_height

    while True:
        x = max_height * np.tan(angle)
        if (2 * x) < frame_width - 100:
            top_left = [0 + x, 0]
            top_right = [frame_width - x, 0]
            bottom_left = [0, frame_height]
            bottom_right = [frame_width, frame_height]
            points = top_right, top_left, bottom_left, bottom_right
            break
        else:
            max_height -= 10
            frame_height -= 10
    return points



def imu_reader_thread():
    """
    Reads the IMU and then caluculates the correct angle for trapezoid
    updates the global imu_angle that is used in the trapezoid calculation
    """
    global imu_angle, current_points, imu, imu_pitch, imu_roll, imu_yaw
    while True:

        imu_pitch, imu_roll, imu_yaw = read_imu(imu)
        imu_angle = homography_angle(VERT_ANGLE, HORIZ_ANGLE, imu_pitch) / np.pi
        time.sleep(0.25)  # Adjust the sleep time as needed


def transformed(frame, points=None):
    """
    This uses homography to transform the current frame for to show app user for the masking process
    
    Args:
    frame (array): current frame
    
    Returns:
    transformed (array): homographied frame
    """
    global MAIN_CONFIG
    dewarped = dewarp_frame(frame)
    with open(MAIN_CONFIG, 'r') as f:
        config = json.load(f)
    try:
        if points:
            trapezoid_points = points
        else:
            trapezoid_points = config.get('trapezoid_points', [])
    except:
        return "No trapezoid points. Please run trapezoid calibrations"
    pts1 = np.float32([
        trapezoid_points[1], trapezoid_points[0], trapezoid_points[2],
        trapezoid_points[3]
    ])
    x_dist = abs(trapezoid_points[2][0] - trapezoid_points[3][0])
    y_dist = abs(trapezoid_points[3][1] - trapezoid_points[1][1])
    pts2 = np.float32([
        [0, 0],
        [x_dist, 0],
        [0, y_dist],
        [x_dist, y_dist],
    ])
    Transform_matrix, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC)
    transformed = cv2.warpPerspective(dewarped, Transform_matrix,
                                      (int(x_dist), int(y_dist)))
    return transformed


  
def get_largest_contour(image):
    """
    Takes the homography image and finds the largest contour as the mask 
    
    Args:
    image (array) homographied image
    
    Returns:
    largest_contour (array): array with lagest contour area
    binary (array): array of image in binary with the larges contour as 1 values
    """

    # Convert to grayscale

    # Apply Otsu's thresholding
    _, binary = cv2.threshold(image, 130, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Find the largest contour
        largest_contour = max(contours, key=cv2.contourArea)
    else:
        largest_contour = []
    return largest_contour, binary


def get_mask_from_largest_contour(image, margin=10):
    """
    Automatic masking process by finding the largest contour area It runs that process twice in order to get a more specific mask
    WARNING: If the image is mainly river and the river has lots of sun spots it will not give a good mask for the whole mask
    
    Args:
    image (array): homographied image for masking
    
    Returns: 
    inner_mask (arrry) final mask generated automatically
    
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    largest_contour, initial_mask = get_largest_contour(gray)
    for i in range(2):  # Adjust the number of iterations as needed
        # Apply the initial mask to the image
        masked_image = cv2.bitwise_and(gray, gray, mask=initial_mask)
        _, binary_masked = cv2.threshold(masked_image, 130, 255,
                                         cv2.THRESH_BINARY)
        contours_masked, _ = cv2.findContours(binary_masked, cv2.RETR_EXTERNAL,
                                              cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest contour within the masked image
        if contours_masked:
            # Find the largest contour
            largest_contour = max(contours_masked, key=cv2.contourArea)
        else:
            largest_contour = []
        # Create a new refined mask
        refined_mask = np.zeros_like(masked_image)
        cv2.drawContours(refined_mask, [largest_contour],
                         -1, (255),
                         thickness=cv2.FILLED)
        initial_mask = refined_mask
    # Erode the mask to shrink it slightly
    kernel = np.ones((margin, margin), np.uint8)
    inner_mask = cv2.erode(initial_mask, kernel, iterations=1)
    return inner_mask


def masked_image(img_path, mask_path, output_path):
    """
    Apply a mask to an image and save the resulting masked image to a specified output path.

    Args:
        img_path (str): The file path to the input image.
        mask_path (str): The file path to the mask image (grayscale).
        output_path (str): The file path where the masked image will be saved.

    Returns:
        None: This function does not return any value. It saves the result to the specified output path.
    """
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    masked_image = cv2.bitwise_and(img, img, mask=mask)
    cv2.imwrite(output_path, masked_image)


def start_imu_thread():
    """
    Start the IMU reader on a separate thread to avoid slowing down video processing.
    
    This function launches a background thread that runs the IMU reader in parallel 
    with the main program to ensure that video processing is not hindered by IMU data collection.
    
    Returns:
        None: This function does not return any value.
    """
    imu_thread = threading.Thread(target=imu_reader_thread)
    imu_thread.daemon = True
    imu_thread.start()


def read_last_config():
    """
    Read the configuration file and check if the device has been calibrated by looking for 
    a 'last_calibrated' key with a valid date.

    Returns:
        str: The value of 'last_calibrated' if present and valid, or None if not found or invalid.
    """
    if os.path.exists(MAIN_CONFIG):
        with open(MAIN_CONFIG, 'r') as file:
            try:
                config_data = json.load(file)
                last_calibrated = config_data.get('last Calibrated', None)
                return last_calibrated if last_calibrated else None
            except json.JSONDecodeError:
                # Handle the case where the file isn't valid JSON
                return None
    return None


def read_last_system_config():
    """
    Read the configuration file and return the site information including the site name, 
    site ID, operator, PIV break, and comments.

    Returns:
        tuple: A tuple containing the site name, site ID, site operator, site PIV break, 
               and site comments. Returns None for missing data or invalid JSON.
    """
    if os.path.exists(MAIN_CONFIG):
        with open(MAIN_CONFIG, 'r') as file:
            try:
                config_data = json.load(file)
                site_name = config_data.get('site_name', None)
                site_id = config_data.get('site_id', None)
                site_operator = config_data.get('site_operator', None)
                site_piv_break = config_data.get('site_piv_break', 15)
                site_comments = config_data.get('site_comments', None)

            except json.JSONDecodeError:
                # Handle the case where the file isn't valid JSON
                return None
    return site_name, site_id, site_operator, site_piv_break, site_comments


def determine_step(count: int):
    """
    Determine the appropriate step value based on the given count.

    Args:
        count (int): The count value used to determine the step.

    Returns:
        int: The determined step value based on the count.
    """
    if count < 10:
        return 1
    elif count < 50:
        return 2
    elif count <= 100:
        return 5
    else:
        return 10


def determine_scale(largest: float):
    """
    Determine the appropriate scale value based on the largest input value.

    Args:
        largest (float): The largest value used to determine the scale.

    Returns:
        int: The determined scale value based on the largest input value.
    """
    if largest < 0.5:
        return 1
    elif largest < 1:
        return 5
    elif largest < 1.5:
        return 10
    elif largest < 2:
        return 20
    else:
        return 40


def plot_vectors_image(data: dict, uScale: np.ndarray, vScale: np.ndarray,
                       magScale: np.ndarray, image: np.ndarray, unit: str):
    """
    Plot the velocity vector field on an image and save the result to a file.

    Args:
        data (dict): A dictionary containing 'xPiv' and 'yPiv' for positions.
        uScale (np.ndarray): The scaled horizontal velocity components.
        vScale (np.ndarray): The scaled vertical velocity components.
        magScale (np.ndarray): The magnitude of the velocity vectors.
        image (np.ndarray): The image to overlay the velocity vectors on.
        unit (str): The unit of velocity ('m/s' or 'fps') to display on the plot.

    Returns:
        None: This function does not return any value. It saves the plot as an image.
    """
    unit_label = 'm/s'
    plt.rcParams['font.size'] = 20
    plt.figure(figsize=(20, 16))
    plt.imshow(image)
    plt.gca().set_xlabel('X')
    plt.gca().set_ylabel('Y')
    if unit == 'fps':
        unit_label = 'ft/s'
    plt.gca().set_title(f'Velocity Vector Field {unit_label}')

    rows_with_non_nan = np.any(~np.isnan(uScale), axis=1)
    count = np.sum(rows_with_non_nan)
    step = determine_step(count)

    largest = np.max(magScale.ravel())
    scalev = determine_scale(largest)

    plt.quiver(data['xPiv'][::step],
               data['yPiv'][::step],
               uScale[::step],
               -vScale[::step],
               magScale[::step],
               scale=scalev,
               pivot='mid',
               cmap='jet')
    cbar = plt.colorbar(fraction=0.016, pad=0.04)
    cbar.ax.tick_params(labelsize=20)
    plt.savefig(os.path.join(BASE_DIR,
                             f'app/static/processed_data/graph_{unit}.jpg'),
                bbox_inches='tight')
    plt.close()


def plot_vectors_mag(data: dict, uScale: np.ndarray, vScale: np.ndarray,
                     magScale: np.ndarray, unit: str):
    """
    Plot the magnitude of velocity vectors and save the result to a file.

    Args:
        data (dict): A dictionary containing 'xPiv' and 'yPiv' for positions.
        uScale (np.ndarray): The scaled horizontal velocity components.
        vScale (np.ndarray): The scaled vertical velocity components.
        magScale (np.ndarray): The magnitude of the velocity vectors.
        unit (str): The unit of velocity ('m/s' or 'fps') to display on the plot.

    Returns:
        None: This function does not return any value. It saves the plot as an image.
    """
    unit_label = 'm/s'
    extent = [0, data['maskPoly'].shape[1], 0, data['maskPoly'].shape[0]]

    plt.rcParams['font.size'] = 20
    plt.figure(figsize=(20, 16))
    plt.imshow(magScale, extent=extent, origin='lower', cmap='jet')
    cbar = plt.colorbar(fraction=0.016, pad=0.04)
    cbar.ax.tick_params(labelsize=20)
    plt.gca().set_xlabel('X')
    plt.gca().set_ylabel('Y')
    if unit == 'fps':
        unit_label = 'ft/s'
    plt.gca().set_title(f'Velocity Vector Field {unit_label}')

    rows_with_non_nan = np.any(~np.isnan(uScale), axis=1)
    count = np.sum(rows_with_non_nan)
    step = determine_step(count)

    largest = np.max(magScale.ravel())
    scalev = determine_scale(largest)

    plt.quiver(data['xPiv'][::step],
               data['yPiv'][::step],
               uScale[::step],
               vScale[::step],
               scale=scalev,
               color='black',
               pivot='mid')
    plt.savefig(os.path.join(
        BASE_DIR, f'app/static/processed_data/graph_{unit}_vectors.jpg'),
                bbox_inches='tight')
    plt.close()


def create_folder():
    """
    Create a new folder with a timestamp, add an entry to the configuration file, 
    and copy necessary files into the new folder.

    Returns:
        None: This function does not return any value.
    """
    global SAVE_CONFIG, MAIN_CONFIG
    # Generate a timestamp
    timestamp = datetime.now().strftime("%Y_%m_%d")

    new_folder_path = os.path.join(SAVE_FOLDER_PATH, timestamp)
    index = 1
    while os.path.exists(new_folder_path):
        new_folder_path = os.path.join(SAVE_FOLDER_PATH,
                                       f'{timestamp}_{index:03d}')
        index += 1
    os.makedirs(new_folder_path, exist_ok=True)

    # Load the config.json
    with open(SAVE_CONFIG, 'r') as f:
        config = json.load(f)

    # Add the new folder entry to config.json
    config['config_folder'] = new_folder_path

    # Save the updated config.json
    with open(SAVE_CONFIG, 'w') as f:
        json.dump(config, f, indent=4)

    shutil.copy(MAIN_CONFIG, os.path.join(new_folder_path, 'config.json'))
    with open(MAIN_CONFIG, 'r') as cf:
        config = json.load(cf)
    try:
        mask_path = config.get('mask_path')
        full_path = os.path.join(BASE_DIR, 'app', mask_path)
        shutil.copy(full_path, os.path.join(new_folder_path, 'mask.jpg'))
    except:
        print('No mask found')


def get_available_disk_space():
    """
    Get the available disk space in MB.

    Returns:
        float: The available disk space in MB.
    """
    statvfs = os.statvfs('/')
    available_space_mb = (statvfs.f_frsize * statvfs.f_bavail) / (1024 * 1024)
    return available_space_mb


def find_usb_device():
    """
    Find the mounted USB device on the system.

    Returns:
        str: The mount point of the USB device, or None if no device is found.
    """
    temp_file = '/tmp/lsblk_output.txt'
    for user in pwd.getpwall():
        if user.pw_uid >= 1000:
            user = user.pw_name

    os.system(f'lsblk -o NAME,SIZE,TYPE,MOUNTPOINT > {temp_file}')
    try:
        # Read the output from the file
        with open(temp_file, 'r') as file:
            for line in file.readlines()[1:]:
                columns = line.split()
                if len(columns) >= 3:
                    name = columns[0]
                    type_ = columns[2]
                    # mountpoint = columns[3] if len(columns) > 3 else ''
                    mountpoint = columns[3:] if len(columns) > 3 else ''
                    if isinstance(mountpoint, list):
                        mountpoint = ' '.join(mountpoint)

                    if type_ == 'part' and mountpoint.startswith(
                            f"/media/{user}"):
                        return mountpoint
    finally:
        # Remove the temporary file
        os.remove(temp_file)

    return None


def mount_and_save_data(save_folder_path: str,
                        usb_mount_point: str = '/media/spec'):
    """
    Mounts a USB device, creates a new directory on it with the current date, 
    and copies necessary directories and files into that directory.

    Args:
        save_folder_path (str): The path to the folder containing the data to be saved.
        usb_mount_point (str): The mount point of the USB device. Default is '/media/spec'.

    Returns:
        tuple: A tuple containing a success or error message and the corresponding HTTP status code.
    """
    mountpoint = find_usb_device()
    print(mountpoint)
    if mountpoint is None:
        return jsonify({"error": "USB drive is not detected."}), 400


    timestamp = datetime.now().strftime("%Y_%m_%d")
    new_directory = os.path.join(mountpoint, f"piv_data_{timestamp}")
    print("new_directory = ", new_directory)

    if os.path.exists(new_directory):
        counter = 1
        while os.path.exists(f"{new_directory}_{counter}"):
            counter += 1
        new_directory = f"{new_directory}_{counter}"
        print ("this is in the if statement ", new_directory)
        os.makedirs(new_directory)
        print("Directory created: ", new_directory)
    else:
        os.makedirs(new_directory, exist_ok=True)

    source = f"{BASE_DIR}/Offsite_Processing"
    dest = f"{new_directory}/Offsite_Processing"

    if not os.path.exists(dest):
        shutil.copytree(source, dest)
    else:
        print("Destination directory already exists. Skipping")

    try:
        os.system(f'sudo chmod -R 777 {save_folder_path}')

        for item in os.listdir(save_folder_path):
            full_item_path = os.path.join(save_folder_path, item)

            if os.path.isdir(full_item_path):
                
                save_usb_log = os.path.join(BASE_DIR,'saving_to_usb.log')
                with open(save_usb_log, 'a') as file:  
                    print("Copying into new directory on USB:", full_item_path, file=file)
                    
                shutil.copytree(full_item_path,
                                os.path.join(new_directory, item))

        dir_output = subprocess.run(['ls', '-l', new_directory],
                                    capture_output=True,
                                    text=True).stdout
        dir_list = []
        for line in dir_output.splitlines()[1:]:
            parts = line.split(maxsplit=8)
            size = parts[4]
            date = ' '.join(parts[5:8])
            name = parts[8]
            dir_list.append({'size': size, 'date': date, 'name': name})
        return jsonify({
            "message": "Directories have been copied to the USB drive successfully.",
            "dir_list": dir_list,
            "new_directory": new_directory
        }), 200
    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


def sort_key(name: str):
    """
    Extracts the date and numeric parts from a directory name for sorting purposes.

    Args:
        name (str): The directory name to be sorted.

    Returns:
        tuple: A tuple containing the date part and the test number (if present) for sorting.
    """
    match = re.match(r"(\d{4}-\d{2}-\d{2})(?:_test(\d+))?", name)
    if match:
        date_part = match.group(0)
        test_part = int(match.group(1)) if match.group(1) else 0
        print(test_part)
        return date_part, test_part
    return name, 0


def checks():
    """
    Checks if the mask and image sizes match, and if not, provides relevant error messages.

    Returns:
        str: A message indicating whether the mask and image are compatible or not.
    """
    global video_handler
    try:
        frame = video_handler.get_frame()
    except:
        video_handler = VideoStreamHandler()
        frame = video_handler.get_frame()

    image = transformed(frame)
    with open(MAIN_CONFIG, 'r') as cf:
        config = json.load(cf)

    try:
        mask_path = config.get('mask_path')
        full_path = os.path.join(BASE_DIR, 'app', mask_path)
        mask = cv2.imread(full_path)
    except:
        if config.get('mask') == 'yes':
            return 'No mask was found and Mask = yes, either go back and Create a mask or change Mask in PIV PARAMETERS to NO'
        else:
            return 'True'

    if mask.shape == image.shape:
        return 'True'
    else:
        return 'Mask shape does not match transformed image shape, please redo mask'


def update_camera_parameters(config):
    """
    Updates global camera parameters based on the provided configuration dictionary.

    Args:
        config (dict): A dictionary containing camera configuration values, including sensor width, 
                       sensor height, focal length, and image resolution.

    Returns:
        None: This function does not return a value but updates global camera parameters.
    """
    global SENSOR_WIDTH, SENSOR_HEIGHT,FOCAL_LENGTH, WIDTH, HEIGHT, VERT_ANGLE, HORIZ_ANGLE,video_handler

    SENSOR_WIDTH = float(config["cmos_sensor_width"])
    SENSOR_HEIGHT = float(config["cmos_sensor_height"])
    FOCAL_LENGTH = float(config["focal_length"])
    WIDTH = int(config["reduced_image_width"])
    HEIGHT = int(config["reduced_image_height"])

    VERT_ANGLE, HORIZ_ANGLE = get_vert_horz_angle(SENSOR_WIDTH, SENSOR_HEIGHT,
                                              FOCAL_LENGTH)
    
    try:
        video_handler.stop()
    except:
        print('video never started')
        
    video_handler=VideoStreamHandler()
    
def update_save_json(test_size):
    """Update save.json with TEST_SIZE while keeping existing data."""
    if os.path.exists(SAVE_CONFIG):
        with open(SAVE_CONFIG, "r") as f:
            data = json.load(f)
    else:
        data = {}  # If the file does not exist, start with an empty dictionary

    # Update or add the test_size field
    data["test_size"] = test_size

    # Write the updated data back to save.json
    with open(SAVE_CONFIG, "w") as f:
        json.dump(data, f, indent=4)
    
"""
Below are all the app and webpage logic. For most there should be a corresponding .html in the templates file that the user will view
"""


@app.after_request
def apply_cors(response: Response):
    """
    Adds CORS headers to the response to allow cross-origin requests.
    """
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers[
        "Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers[
        "Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    return response


@app.before_request
def check_process():
    """
    Checks if the process is running before each request.

    This function sets the `process_running` flag in the `g` object to check the status
    of the process, ensuring that the system is ready to handle requests.
    """
    is_process_running()


@login_manager.user_loader
def load_user(username: str):
    """
    Reloads the user object from the user ID stored in the session.
    """
    if username == USERNAME:
        return User(username)
    return None


@app.route('/', methods=['GET', 'POST'])
def index() -> Response:
    """
    Renders the login page.

    This is the default route that renders the login page of the captive portal.

    """
    return render_template('login.html')


@app.route('/login', methods=['GET', 'POST'])
def login() -> Response:
    """
    Handles user login. Authenticates the user and redirects based on the process state.

    This route processes the login form, authenticates the credentials, and checks the
    state of the PIV process. If the process is running, the user is redirected to the
    PIV options; otherwise, they are redirected to the calibration splash page.

    """
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        # Authenticate user
        if username == USERNAME and password == PASSWORD:
            user = User(username)
            login_user(user)

            # Check the PIV process state
            with open(monitor_file_path, 'r') as f:
                content = f.read().strip()

            if content == "run":
                return redirect(url_for('running_piv_options'))
            else:
                global video_handler
                try:
                    video_handler = VideoStreamHandler()
                except:
                    print('Video already on')
                return redirect(url_for('calibrate_splash'))
        else:
            return render_template('login.html',
                                   error_message="Invalid credentials")

    return render_template('login.html')


@app.after_request
def after_request(response: Response):
    """
    Adds a custom header to the response to indicate that the device is on a captive portal login page.
    
    """
    response.headers['Captive-Portal-Login'] = 'true'
    return response


@app.route('/splash')
@login_required
def splash():
    """
    The main menu with all the options a user can select
    """
    error = request.args.get('error')
    return render_template('splash.html', error=error)


@app.route('/splash_utilities')
@login_required
def splash_utilities():
    """
    The main menu with only the utilities a user can select.
    Trying to clean up the main splash page.
    """
    return render_template('splash_utilities.html')


@app.route('/splash_calib_setup')
@login_required
def splash_calib_setup():
    """
    The main menu with only the utilities a user can select.
    Trying to clean up the main splash page.
    """
    return render_template('splash_calib_setup.html')


@app.route('/splash_piv')
@login_required
def splash_piv():
    """
    The main menu with only the utilities a user can select.
    Trying to clean up the main splash page.
    """
    return render_template('splash_piv.html')


@app.route('/trapezoid')
@login_required
def trapezoid():
    """
    Displays the trapezoid calibration page
    """
    return render_template('trapezoid.html')


@app.route('/piv_parameters')
@login_required
def piv_parameters():
    """
    Displays the PIV params page, it also checks if the config file has values and displays those as the default (no it doesn't yet)
    """
    global MAIN_CONFIG
    if os.path.exists(MAIN_CONFIG):
        with open(MAIN_CONFIG, 'r') as file:
            config_data = json.load(file)
    else:
        config_data = {}

    return render_template('piv_parameters.html', config=config_data)


@app.route('/save_config', methods=['POST'])
@login_required
def save_config():
    """
    Saves PIV parameters to the config file
    """
    global MAIN_CONFIG
    data = request.json
    try:
        try:
            with open(MAIN_CONFIG, 'r') as f:
                config = json.load(f)
        except Exception as e:
            config = {}

        for key, value in data.items():
            config[key] = value
        with open(MAIN_CONFIG, 'w') as f:
            json.dump(config, f, indent=4)
        update_camera_parameters(config)
        return jsonify({"message": "Configuration saved successfully"}), 200


    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/camera_parameters')
@login_required
def camera_parameters():
    """
    Displays the camera parameters page, it also checks if the config file has values and displays those as the default 
    """
    global MAIN_CONFIG
    if os.path.exists(MAIN_CONFIG):
        with open(MAIN_CONFIG, 'r') as file:
            config_data = json.load(file)
    else:
        config_data = {}

    return render_template('camera_parameters.html', config=config_data)


@app.route('/site_info')
@login_required
def site_info():
    """
    Displays system configuration page with existing values and ability to change them
    """
    sys_config = read_last_system_config()

    return render_template('site_info.html',
                           site_name=sys_config[0],
                           site_id=sys_config[1],
                           site_operator=sys_config[2],
                           site_piv_break=sys_config[3],
                           site_comments=sys_config[4])


@app.route('/results')
@login_required
def results():
    """
    Displays all Config options for user to choose from
    """
    config_dirs = sorted(os.listdir(SAVE_FOLDER_PATH),
                         key=sort_key,
                         reverse=True)
    print(config_dirs)
    return render_template('results.html', config_dirs=config_dirs)


@app.route('/results/get_runs', methods=['POST'])
@login_required
def get_runs():
    """
    Get avalible PIV runs based on user specified config
    """
    config_dir = request.json.get('config_dir')
    config_path = os.path.join(SAVE_FOLDER_PATH, config_dir)

    print("Checking directory:", config_path)

    if os.path.exists(config_path):
        runs = sorted([
            d for d in os.listdir(config_path)
            if os.path.isdir(os.path.join(config_path, d))
        ],
                      reverse=True)
        print("Available runs:", runs)
        return jsonify({"runs": runs})
    else:
        print("Configuration not found:", config_dir)
        return jsonify({"error": "Configuration not found."}), 404



@app.route('/results/graph_data', methods=['GET'])
@login_required
def graph_data():
    """
    Graph data from user specified PIV run
    """
    config_dir = request.args.get('config_dir')
    run = request.args.get('run')
    unit = request.args.get('unit')
    display_type = request.args.get('display_type')

    config_path = os.path.join(SAVE_FOLDER_PATH, config_dir, run)
    latest_file = os.path.join(
        config_path, f"{run}_PIV_output")

    if os.path.exists(latest_file):
        data = process_data(latest_file)
        magScale = data['magScale']

        if unit == 'fps':
            magScale *= 3.28084  # Convert to fps
            uScale = data['uScale'] * 3.28084
            vScale = data['vScale'] * 3.28084
        else:
            uScale = data['uScale']
            vScale = data['vScale']

        if display_type == 'image':
            print(os.path.join(config_path, 'captured_image.jpg'))
            image = cv2.imread(os.path.join(config_path, 'capture_image.jpg'))
            plot_vectors_image(data, uScale, vScale, magScale, image, unit)
            time.sleep(1)
            return jsonify({"url": f'/static/processed_data/graph_{unit}.jpg'})
        elif display_type == 'magnitude':
            plot_vectors_mag(data, uScale, vScale, magScale, unit)
            time.sleep(1)
            return jsonify(
                {"url": f'/static/processed_data/graph_{unit}_vectors.jpg'})
    else:
        return jsonify({"error": "Data file not found."}), 404


@app.route('/results/get_csv_data', methods=['POST'])
@login_required
def get_csv_data():
    """
    Gets the CSV data to be displayed for the user specified PIV run
    """
    config_dir = request.json.get('config_dir')
    run = request.json.get('run')
    filename = request.json.get('filename')

    config_path = os.path.join(SAVE_FOLDER_PATH, config_dir, run)
    file_path = os.path.join(config_path, f"{run}_PIV_output",
                             f"{run}_{filename}.csv")

    print("Looking for CSV file at:", file_path)

    if os.path.exists(file_path):
        df = pd.read_csv(file_path)

        df = df.fillna('-')
        return df.to_html(index=False, header=False)
    else:
        if filename == "magScale_fps":
            file_path = os.path.join(config_path, f"{run}_PIV_output",
                                     f"{run}_magScale.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df = df * 3.28084
                return df.to_html(index=False, header=False)
        elif filename=="uScale_fps":
            file_path=os.path.join(config_path,f"{run}_PIV_output", f"{run}_uScale.csv")
            if os.path.exists(file_path):
                df=pd.read_csv(file_path)
                df = df*3.28084
                return df.to_html(index=False, header=False)
        elif filename=="vScale_fps":
            file_path=os.path.join(config_path,f"{run}_PIV_output", f"{run}_vScale.csv")
            if os.path.exists(file_path):
                df=pd.read_csv(file_path)
                df = df*3.28084
                return df.to_html(index=False, header=False)
        print("CSV file not found:", file_path)
        return "File not found", 404


@app.route('/video_feed')
@login_required
def video_feed():
    """
    Displays the live feed with a back button
    """

    def generate_frames():
        global video_handler
        while True:
            try:
                frame = video_handler.get_frame()
            except:
                try:
                    video_handler = VideoStreamHandler()
                    frame = video_handler.get_frame()
                except:
                    print('something is wrong')
            frame = dewarp_frame(frame)
            if frame is not None:
                ret, jpeg = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() +
                       b'\r\n')
            time.sleep(0.1)

    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/live_feed')
@login_required
def live_feed():
    """
    Renders the HTML page with live video
    """
    return render_template('live_feed.html')


@app.route('/process_trapezoid')
@login_required
def process_trapezoid():
    """
    The trapezoid live video process runs through this and is displayed on the trapezoid page
    """

    def generate_frames():
        global current_points, imu_angle
        global video_handler

        while True:
            try:
                frame = video_handler.get_frame()
            except:
                try:
                    video_handler = VideoStreamHandler()
                    frame = video_handler.get_frame()
                except:
                    print('something is wrong')
            dewarped_frame = dewarp_frame(frame)

            if current_points == None:
                current_points = find_largest_trapezoid(angle=imu_angle)

            get_trapezoid(imu_angle)

            if current_points is not None:
                cv2.polylines(dewarped_frame, [
                    np.array(current_points, dtype=np.int32).reshape((-1, 1, 2))
                ],
                              isClosed=True,
                              color=(0, 255, 0),
                              thickness=15)

                _, jpeg = cv2.imencode('.jpg', dewarped_frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() +
                       b'\r\n')

    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/slide_point', methods=['POST'])
@login_required
def slide_point():
    """
    The logic for moving the trapezoid on the trapezoid points with sliders
    """
    global current_points, imu_angle
    point = request.args.get('point')
    value = request.args.get('value')

    print(f"Attempting to move {point} {value}")

    if current_points is not None:
        if point == 'top_left_y':
            current_points[1][1] = int(value)
        elif point == 'top_left_x':
            current_points[1][0] = int(value)
        elif point == 'bottom_left_y':
            current_points[2][1] = int(value)
        elif point == 'top_right_x':
            current_points[0][0] = int(value)
        elif point == 'bottom_left_y':
            current_points[2][1] = int(value)

        get_trapezoid(imu_angle)
    else:
        print("current_points is None")
    return jsonify({"status": "OK"})


@app.route('/save_points', methods=['POST'])
@login_required
def save_points():
    """
    Saves the current trapezoid points to the config file when user selects save buttom
    """
    global MAIN_CONFIG
    try:
        try:
            with open(MAIN_CONFIG, 'r') as f:
                config = json.load(f)
        except Exception as e:
            config = {}
        config['trapezoid_points'] = [point for point in current_points]
        with open(MAIN_CONFIG, 'w') as f:
            json.dump(config, f, indent=4)

        return 'Points saved successfully!', 200
    except Exception as e:
        return str(e), 500


@app.route('/transformed_image')
@login_required
def transformed_image():
    """
    Creates a transformed image with the current trapezoid points
    """
    global video_handler
    try:
        frame = video_handler.get_frame()
    except:
        
        print('no frame?')
    if frame is None:
        return jsonify({'error': 'Unable to capture frame from video stream'
                       }), 500

    transformed_frame = transformed(frame, current_points)
    transformed_path = os.path.join(BASE_DIR,
                                    'app/static/mask/captured_frame.jpg')
    cv2.imwrite(transformed_path, transformed_frame)
    return send_file('static/mask/captured_frame.jpg', mimetype='image/png')


@app.route('/masking_options', methods=['GET'])
@login_required
def masking_options():
    """
    Displays the masking options either digitize or generate for a user to decide
    """
    return render_template('masking_options.html')


@app.route('/masking_result', methods=['GET'])
@login_required
def masking_result():
    """
    Displays the masking result with the original image and the corresponding created mask
    """
    original_image = request.args.get('original_image')
    mask_image = request.args.get('mask_image')
    if not original_image or not mask_image:
        return "Error: Missing image paths", 400

    return render_template('masking_result.html',
                           original_image=original_image,
                           mask_image=mask_image,
                           masked_image='static/mask/masked_image.jpg')


@app.route('/save_mask_path', methods=['POST'])
@login_required
def save_mask_path():
    """
    Saves the user selected mask path
    """
    data = request.get_json()
    mask_path = data.get('mask_path')

    if mask_path:
        with open(MAIN_CONFIG, 'r') as f:
            config = json.load(f)


        config['mask_path'] = mask_path

        with open(MAIN_CONFIG, 'w') as f:
            json.dump(config, f, indent=4)

        return jsonify({"status": "success"}), 200
    else:
        return jsonify({
            "status": "error",
            "message": "Mask path is missing"
        }), 400


@app.route('/generate_mask', methods=['GET', 'POST'])
@login_required
def generate_mask():
    """
    Logic for running the automatic masking process
    """
    global video_handler
    if request.method == 'POST':

        try:
            frame = video_handler.get_frame()
        except:
            try:
                video_handler = VideoStreamHandler()
                frame = video_handler.get_frame()
            except:
                print('somthing is wrong')
        transformed_frame = transformed(frame)

        mask = get_mask_from_largest_contour(transformed_frame)

        original_image_path = os.path.join(
            BASE_DIR, 'app/static/mask/captured_frame.jpg')
        mask_path = os.path.join(BASE_DIR, 'app/static/mask/mask_generated.png')
        cv2.imwrite(original_image_path, transformed_frame)
        cv2.imwrite(mask_path, mask)
        output_path = os.path.join(BASE_DIR, 'app/static/mask/masked_image.jpg')
        masked_image(original_image_path, mask_path, output_path)

        return render_template('generate_mask.html',
                               original_image='static/mask/captured_frame.jpg',
                               mask_image='static/mask/mask_generated.png')

    else:
        return render_template('generate_mask.html')


@app.route('/digitize_mask', methods=['GET', 'POST'])
@login_required
def digitize_mask():
    global video_handler
    """
    Logic for the digitize (user selects points on image and creats a mask that way) process
    """
    if request.method == 'POST':
        data = request.json
        points = data.get('points')

        if not points:
            return jsonify({'error': 'No points received'}), 400

        try:
            frame = video_handler.get_frame()
        except:
            try:
                video_handler = VideoStreamHandler()
                frame = video_handler.get_frame()
            except:
                print('something is wrong')
        if frame is None:
            return jsonify(
                {'error': 'Unable to capture frame from video stream'}), 500

        transformed_frame = transformed(frame)
        points = np.array(points, dtype=np.int32)
        mask = np.zeros_like(transformed_frame)
        cv2.fillPoly(mask, [points], (255, 255, 255))

        original_image_path = os.path.join(
            BASE_DIR, 'app/static/mask/captured_frame.jpg')
        mask_path = os.path.join(BASE_DIR, 'app/static/mask/mask_digitized.png')
        cv2.imwrite(original_image_path, transformed_frame)
        cv2.imwrite(mask_path, mask)
        output_path = os.path.join(BASE_DIR, 'app/static/mask/masked_image.jpg')
        masked_image(original_image_path, mask_path, output_path)

        return redirect(
            url_for('masking_result',
                    original_image='static/mask/captured_frame.jpg',
                    mask_image='static/mask/mask_digitized.png'))

    else:

        try:
            frame = video_handler.get_frame()
        except:
            try:
                video_handler = VideoStreamHandler()
                frame = video_handler.get_frame()
            except:
                print('somthing is wrong')

        if frame is None:
            return "Error: Unable to capture frame from video stream", 500

        transformed_frame = transformed(frame)
        frame_path = os.path.join(BASE_DIR,
                                  'app/static/mask/captured_frame.jpg')
        cv2.imwrite(frame_path, transformed_frame)

        return render_template('digitize_mask.html',
                               image_path='static/mask/captured_frame.jpg')


@app.route('/calibrate_splash')
@login_required
def calibrate_splash():
    """
    Displays if the device has been calibrated or not and give the user the option to calibrate or not
    """
    print("Calibrate splash route accessed", flush=True)
    last_config = read_last_config()
    print('last_config = ', last_config)

    if not last_config:
        message = "Device needs to be Calibrated"
    else:
        message = f"Last Calibration: {last_config}"

    return render_template('calibrate_splash.html', message=message)


@app.route('/calibrate_piv_parameters')
@login_required
def calibrate_piv_parameters():
    """
    The same as the other PIV_parameter page but has a next button guiding the user through the calibration process
    """
    global MAIN_CONFIG
    if os.path.exists(MAIN_CONFIG):
        with open(MAIN_CONFIG, 'r') as file:
            config_data = json.load(file)
    else:
        config_data = {}

    return render_template('calibrate_piv_parameters.html', config=config_data)


@app.route('/calibrate_trapezoid')
@login_required
def calibrate_trapezoid():
    """
    The same as the other Trapezoid page but has a next button guiding the user through the calibration process
    """
    return render_template('calibrate_trapezoid.html')


@app.route('/calibrate_masking_options', methods=['GET'])
@login_required
def calibrate_masking_options():
    """
    The same as the other masking options page but has a next button guiding the user through the calibration process
    """
    return render_template('calibrate_masking_options.html')


@app.route('/calibrate_masking_result', methods=['GET'])
@login_required
def calibrate_masking_result():
    """
    The same as the other masking result page but has a test button to test the calibrated system
    """
    original_image = request.args.get('original_image')
    mask_image = request.args.get('mask_image')

    if not original_image or not mask_image:
        return "Error: Missing image paths", 400

    return render_template('calibrate_masking_result.html',
                           original_image=original_image,
                           mask_image=mask_image,
                           masked_image='static/mask/masked_image.jpg')


@app.route('/calibrate_generate_mask', methods=['GET', 'POST'])
@login_required
def calibrate_generate_mask():
    global video_handler
    """
    The same as the other generate_mask page
    """
    if request.method == 'POST':
        try:
            frame = video_handler.get_frame()
        except:
            try:
                video_handler = VideoStreamHandler()
                frame = video_handler.get_frame()
            except:
                print('something is wrong')
        transformed_frame = transformed(frame)

        mask = get_mask_from_largest_contour(transformed_frame)

        original_image_path = os.path.join(
            BASE_DIR, 'app/static/mask/captured_frame.jpg')
        mask_path = os.path.join(BASE_DIR, 'app/static/mask/mask_generated.png')
        cv2.imwrite(original_image_path, transformed_frame)
        cv2.imwrite(mask_path, mask)
        output_path = os.path.join(BASE_DIR, 'app/static/mask/masked_image.jpg')
        masked_image(original_image_path, mask_path, output_path)

        return render_template('calibrate_generate_mask.html',
                               original_image='static/mask/captured_frame.jpg',
                               mask_image='static/mask/mask_generated.png')

    else:
        return render_template('calibrate_generate_mask.html')


@app.route('/calibrate_digitize_mask', methods=['GET', 'POST'])
@login_required
def calibrate_digitize_mask():
    global video_handler
    """
    The same as the other digitize_mask page but has a next button guiding the user through the calibration process
    """
    if request.method == 'POST':
        data = request.json
        points = data.get('points')

        if not points:
            return jsonify({'error': 'No points received'}), 400

        try:
            frame = video_handler.get_frame()
        except:
            try:
                video_handler = VideoStreamHandler()
                frame = video_handler.get_frame()
            except:
                print('somthing is wrong')
        if frame is None:
            return jsonify(
                {'error': 'Unable to capture frame from video stream'}), 500

        transformed_frame = transformed(frame)
        points = np.array(points, dtype=np.int32)
        mask = np.zeros_like(transformed_frame)
        cv2.fillPoly(mask, [points], (255, 255, 255))

        original_image_path = os.path.join(
            BASE_DIR, 'app/static/mask/captured_frame.jpg')
        mask_path = os.path.join(BASE_DIR, 'app/static/mask/mask_digitized.png')
        cv2.imwrite(original_image_path, transformed_frame)
        cv2.imwrite(mask_path, mask)
        output_path = os.path.join(BASE_DIR, 'app/static/mask/masked_image.jpg')
        masked_image(original_image_path, mask_path, output_path)

        return redirect(
            url_for('calibrate_masking_result',
                    original_image='static/mask/captured_frame.jpg',
                    mask_image='static/mask/mask_digitized.png'))

    else:
        try:
            frame = video_handler.get_frame()
        except:
            try:
                video_handler = VideoStreamHandler()
                frame = video_handler.get_frame()
            except:
                print('somthing is wrong')
        if frame is None:
            return "Error: Unable to capture frame from video stream", 500

        transformed_frame = transformed(frame)
        frame_path = os.path.join(BASE_DIR,
                                  'app/static/mask/captured_frame.jpg')
        cv2.imwrite(frame_path, transformed_frame)

        return render_template('calibrate_digitize_mask.html',
                               image_path='static/mask/captured_frame.jpg')


@app.route('/stream')
def stream():
    '''
    Stream log file updates to the client.
    This implementation clears the log file first, then reads new updates with timeouts and error handling.
    '''
    test_LOG = f'{BASE_DIR}/script.log'

    try:
        if os.path.exists(test_LOG):
            with open(test_LOG, 'w') as f:
                pass 
    except Exception as e:
        print(f"Error clearing log file: {str(e)}")

    def generate():
        last_position = 0

        try:
            while True:
                if not os.path.exists(test_LOG):
                    yield f"data: Waiting for log file to be created...\n\n"
                    time.sleep(1)
                    continue

                try:
                    with open(test_LOG, "r") as log_file:
                        log_file.seek(last_position)

                        new_lines = log_file.readlines()
                        if new_lines:
                            for line in new_lines:
                                yield f"data: {line.strip()}\n\n"
                            last_position = log_file.tell()
                        else:
                            yield f"data: ...\n\n"

                except (IOError, OSError) as e:
                    yield f"data: Error reading log file: {str(e)}\n\n"

                time.sleep(0.5)

        except GeneratorExit:
            pass
        except Exception as e:
            yield f"data: Stream error: {str(e)}\n\n"

    response = Response(stream_with_context(generate()), content_type='text/event-stream')
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['X-Accel-Buffering'] = 'no'
    return response


@app.route('/test')
@login_required
def run_script():
    """
    Redirects users to the waiting html
    """

    return render_template('waiting.html')


@app.route('/run_test_script')
@login_required
def run_test_script():
    """
    Stops the video (so gstreamer does not freak out with multiple scripts trying to access it) and runs the test script for a 5 second test
    the user is then redirect to the results page to view the results
    """
    check_result = checks()


    if check_result != "True":
        return redirect(url_for('splash', error=check_result))
    global video_handler

    try:
        video_handler.stop()
    except:
        print('no video running')

    timestamp = datetime.now().strftime("%Y_%m_%d")
    index = 1
    new_folder_path = os.path.join(SAVE_FOLDER_PATH,
                                   f'{timestamp}_test00{index}')

    while os.path.exists(new_folder_path):
        logging.info(f'path {new_folder_path} exists, incrementing')
        if index < 10:
            new_folder_path = os.path.join(SAVE_FOLDER_PATH,
                                           f'{timestamp}_test00{index}')
        else:
            new_folder_path = os.path.join(SAVE_FOLDER_PATH,
                                           f'{timestamp}_test0{index}')
        index += 1

    logging.info(f'creating new folder {new_folder_path}')
    os.makedirs(new_folder_path, exist_ok=True)

    try:
        with open(SAVE_CONFIG, 'r') as f:
            save_config = json.load(f)

    except Exception as e:
        logging.error(f'unable to load {SAVE_CONFIG}')
        sys.exit(1)

    save_config['config_folder'] = new_folder_path
    with open(MAIN_CONFIG, 'r') as cf:
        config = json.load(cf)
    try:
        mask_path = config.get('mask_path')
        full_path = os.path.join(BASE_DIR, 'app', mask_path)
        shutil.copy(full_path, os.path.join(new_folder_path, 'mask.jpg'))
    except:
        print('no mask')

    with open(SAVE_CONFIG, 'w') as f:
        json.dump(save_config, f, indent=4)

    shutil.copy(MAIN_CONFIG, os.path.join(new_folder_path, 'config.json'))

    os.system(os.path.join(BASE_DIR, "test_PIV.sh"))
    video_handler = VideoStreamHandler()

    return redirect(url_for('results'))


@app.route('/save_and_run')
@login_required
def save_and_run():
    """
    Displays the save and run page that give the options to run PIV or go back to the splash menu
    """
    global TEST_SIZE
    test_dirs = sorted([d for d in os.listdir(SAVE_FOLDER_PATH) if 'test' in d],
                       reverse=True)
    try:
        most_recent_test = test_dirs[0]
        most_recent_path = os.path.join(SAVE_FOLDER_PATH, most_recent_test)
        file_size = 0
        for dirpath, dirnames, filenames in os.walk(most_recent_path):
            for f in filenames:
                file_size += os.path.getsize(os.path.join(dirpath, f))
        TEST_SIZE = file_size

        # Save TEST_SIZE persistently in save.json
        update_save_json(TEST_SIZE)

    except:
        # Load the last saved TEST_SIZE from save.json if available
        with open(SAVE_CONFIG, "r") as f:
            data = json.load(f)
        TEST_SIZE = data.get("test_size", "No Test")
        
    # Get available disk space
    total_space = shutil.disk_usage("/").free
    free_gb = total_space / (1024**3)

    # Load configuration settings
    with open(MAIN_CONFIG, 'r') as cf:
        config = json.load(cf)
    try:
        length = int(config.get("site_piv_break"))
    except:
        length = 15  # Default to 15 minutes if not found

    if isinstance(TEST_SIZE, int) and TEST_SIZE > 0:
        run_duration_hours = (total_space // TEST_SIZE) * (length / 60)
        run_out_date = (datetime.now() + timedelta(hours=run_duration_hours)).date()
        run_out_date_str = run_out_date.strftime("%m-%d-%y")
        days_left = int(run_duration_hours // 24)  # Convert to days
    else:
        days_left = "N/A"

    return render_template(
        'save_and_run.html',
        total_space_gb=round(free_gb, 2),
        days_left=run_out_date_str
    )


@app.route('/save_data')
@login_required
def save_data():
    """
    Displays the data management page that gives the options to save to usb, delete or go back to the splash menu
    """
    return render_template('save_data.html')


@app.route('/save_data_piv')
@login_required
def save_data_piv():
    """
    Displays the data management page that gives the options to save to usb, delete or go back to the for rn piv
    """
    return render_template('save_data_piv.html')


@app.route('/handle_test_files', methods=['POST'])
@login_required
def handle_test_files():
    """
    Either deletes or saves test files to USB when user is about to run PIV.
    """
    action = request.args.get('action')

    # test_dirs = sorted([d for d in os.listdir(SAVE_FOLDER_PATH) if 'test' in d],
    #                    reverse=True)

    # if not test_dirs:
    #     return jsonify({"message": "No test files found."})

    # most_recent_test = test_dirs[0]
    # most_recent_path = os.path.join(SAVE_FOLDER_PATH, most_recent_test)
    global TEST_SIZE
    test_dirs = sorted([d for d in os.listdir(SAVE_FOLDER_PATH) if 'test' in d],
                       reverse=True)
    total_space = shutil.disk_usage("/").free
    free_gb = total_space / (1024**3)
    print(f"Total free disk space: {total_space} bytes")
    # file_size = 0


    # for dirpath, dirnames, filenames in os.walk(most_recent_path):
    #     for f in filenames:
    #         file_size += os.path.getsize(os.path.join(dirpath, f))
    with open(MAIN_CONFIG, 'r') as cf:
        config = json.load(cf)
    try:
        length = int(config.get("site_piv_break"))
    except:
        length = 15
    if isinstance(TEST_SIZE,int):
        run_duration_hours = (total_space // TEST_SIZE) * length / 60
        run_out_date = datetime.now() + timedelta(hours=run_duration_hours)
        run_out_date_str = run_out_date.strftime("%m-%d-%y")  # %H:%M:%S")
    else:
        run_out_date_str = f'No test files to calculate run out date. Total Space:{free_gb}'
    if action == 'delete':
        for test_dir in test_dirs:
            shutil.rmtree(os.path.join(SAVE_FOLDER_PATH, test_dir))
        return jsonify({
            "message": "All test files deleted.",
            "run_out_date": run_out_date_str
        })

    elif action == 'save':
        response = mount_and_save_data(SAVE_FOLDER_PATH, "/media/usb")
        return response

    return jsonify({"message": "Invalid action."})


@app.route('/run_process', methods=['GET', 'POST'])
@login_required
def run_process():
    """
    Starts the PIV process by first stopping the video and then writing 'run' to a text file that is being monitored by the runAll.sh
    script indicating that it should start the run process. Users are then redirected to an option menu where they can either view the results or 
    cancel piv
    """
    with open(MAIN_CONFIG, 'r') as cf:
        config = json.load(cf)
    config["last Calibrated"] = datetime.now().strftime("%m-%d-%y")
    with open(MAIN_CONFIG, 'w') as f:
        json.dump(config, f, indent=4)
    create_folder()
    # global video_handler
    # video_handler.stop()

    with open(monitor_file_path, 'w') as f:
        f.write("run")

        
    return redirect(url_for('running_piv_options'))


@app.route('/running_piv_options')
@login_required
def running_piv_options():
    """
    Displays options of viewing results or canceling the process
    """

    return render_template('running_piv_options.html')


@app.route('/cancel_piv', methods=['POST'])
@login_required
def cancel_piv():
    """
    Cancels the PIV process by writing "stop" to the monitored process.
    Displays a waiting page if the device is busy and redirects to the splash page when free.
    """
    global video_handler

    with open(monitor_file_path, 'w') as f:
        f.write("stop")

    try:
        video_handler = VideoStreamHandler()
    except:
        print(' video running')


    return render_template('waiting_for_device.html')


@app.route('/check_device_status')
@login_required
def check_device_status():
    """
    Check the device status and return the current state (whether it's free or busy).
    """

    global video_handler
    if video_handler.cap.isOpened():
        time.sleep(2)
        return jsonify({'status': 'free'})
    else:
        return jsonify({'status': 'busy'})


@app.route('/save_to_usb', methods=['POST', 'GET'])
@login_required
def save_to_usb():
    """
    Saved files to USB drive
    """
    save_dir = os.path.join(BASE_DIR, 'save_data')

    print("save_dir = ", save_dir)
    response = mount_and_save_data(save_dir, "/media/usb")
    return response

@app.route('/save_success')
@login_required
def save_success():
    """
    Displays usb directories to confirm saving.
    """
    dir_list = json.loads(request.args.get('dir_list', '[]'))  # Get dir_list from query param
    new_directory = request.args.get('new_directory', '')
    
    return render_template(
        "save_success.html",
        message="Directories have been copied to the USB drive successfully.",
        dir_list=dir_list,
        new_directory=new_directory
    )

@app.route('/logs')
@login_required
def logs_page():
    """
    Renders log page
    """
    return render_template('logs.html')


@app.route('/logs/<log_type>')
@login_required
def view_logs(log_type):
    """
    Displays logs based on user seletion. This will only display the logs starting from the last reboot
    """
    log_files = {
        'app': '/var/log/spec_app.log',
        'piv': '/var/log/piv_process.log',
        'gstreamer': '/var/log/gstreamer.log',
        'loopback': '/var/log/loopback.log',
        'diskSpace': '/var/log/disk_space.log'
    }

    log_file_path = log_files.get(log_type)
    log_content = []

    if log_type == 'app' or log_type == 'piv':
        if not os.path.exists(log_file_path):
            try:
                os.path.exists(f'{log_file_path}.1')
                log_file_path = f'{log_file_path}.1'
            except:
                log_content = 'Error in loading logs'

    if log_file_path and os.path.exists(log_file_path):
        with open(log_file_path, 'r') as f:
            all_lines = f.readlines()

        if log_type == 'app':
            STARTUP_IDENTIFIER = "* Running on http://127.0.0.1:5000"
            log_content = filter_log_content(all_lines, STARTUP_IDENTIFIER)
        elif log_type == 'piv':
            STARTUP_IDENTIFIER_PIV = 'PIV SCRIPT STARTED'
            log_content = filter_log_content(all_lines, STARTUP_IDENTIFIER_PIV)
        elif log_type == 'diskSpace':
            STARTUP_IDENTIFIER_DISK = 'IT IS WORKING'
            log_content = filter_log_content(all_lines, STARTUP_IDENTIFIER_DISK)
        else:
            log_content = all_lines
    else:
        log_content = [f"Log file for '{log_type}' not found."]

    return '\n'.join(log_content)


def filter_log_content(all_lines, startup_identifier):
    try:
        last_startup_index = max(idx for idx, line in enumerate(all_lines)
                                 if startup_identifier in line)
        return all_lines[last_startup_index + 1:]
    except ValueError:
        return all_lines


@app.route('/logout')
@login_required
def logout():
    """
    Logs user out of app
    """
    logout_user()
    return redirect(url_for('login'))


@app.route('/get_current_points', methods=['GET'])
def get_current_points():
    """
    Grabs the current points of the trapezoid
    """
    print("current points = ", current_points)
    return jsonify(current_points)


@app.route('/bubble_level_2')
# @login_required
def bubble_level_2():
    """
    Renders the bubble level
    """
    return render_template('bubble_level_2.html')


@app.route('/read_IMU_for_level', methods=['GET'])
def read_IMU_for_level():
    """
    Returns:
    floats: The pitch angle in radians, adjusted to a coordinate system based on IMU positioning.
    The roll and yaw angles in radians.
    """

    global imu_pitch, imu_roll, imu_yaw
    return ([imu_pitch, imu_roll, imu_yaw])


@app.route('/calibrate_reset_trapezoid')
@login_required
def calibrate_reset_trapezoid():
    """
    Resets the trapezoid to the largest possible one
    """
    global current_points
    current_points = None
    process_trapezoid()
    return render_template('calibrate_trapezoid.html')


@app.route('/reset_trapezoid')
@login_required
def reset_trapezoid():
    """
    Resets the trapezoid to the largest possible one
    """
    global current_points
    current_points = None
    process_trapezoid()
    return render_template('trapezoid.html')


@app.route('/reboot', methods=['POST'])
@login_required
def reboot():
    """
    Reboots the system
    """
    os.system("(sleep 5; sudo reboot) &")
    return "Rebooting in 5 seconds...", 200


@app.route('/select_and_delete')
@login_required
def select_and_delete():
    """
    Gets the list of folders in the save directory
    """

    try:
        folders = [
            name for name in os.listdir(SAVE_FOLDER_PATH)
            if os.path.isdir(os.path.join(SAVE_FOLDER_PATH, name))
        ]
    except Exception as e:
        folders = []
        flash(f"Error reading folders: {e}", 'error')
    return render_template('select_and_delete.html', folders=folders)


@app.route('/delete-folder', methods=['POST'])
def delete_folder():
    """
    Gives the user the option of which files to delete and deletes them
    """
    selected_folders = request.form.getlist('folders[]')

    if not selected_folders:
        flash("No folders selected.", 'error')
        return redirect(url_for('select_and_delete'))

    for folder_name in selected_folders:
        folder_path = os.path.join(SAVE_FOLDER_PATH, folder_name)
        if os.path.isdir(folder_path):
            try:
                shutil.rmtree(folder_path)
                flash(f"Folder '{folder_name}' deleted successfully!",
                      'success')
            except Exception as e:
                flash(f"Error deleting folder '{folder_name}': {e}", 'error')
        else:
            flash(f"Invalid folder: {folder_name}", 'error')

    return redirect(url_for('select_and_delete'))


@app.route('/calibrate_trapezoid_instructions')
@login_required
def calibrate_trapezoid_instructions():
    """
    Provides instructions for trapezoid calibration. This includes setting up the area for PIV calculations.
    """
    return render_template('calibrate_trapezoid_instructions.html')


@app.route('/calibrate_piv_parameters_instructions')
@login_required
def calibrate_piv_parameters_instructions():
    """
    Provides instructions for calibrating PIV parameters. This covers the configuration of critical PIV settings.
    """
    return render_template('calibrate_piv_parameters_instructions.html')


@app.route('/calibrate_masking_instructions')
@login_required
def calibrate_masking_instructions():
    """
    Provides instructions for masking calibration. This guides users on how to exclude non-river areas for PIV calculations.
    """
    return render_template('calibrate_masking_instructions.html')


@app.route('/calibrate_splash_instructions')
@login_required
def calibrate_splash_instructions():
    """
    Displays the splash page for calibration. This serves as an entry point to the calibration steps.
    """
    return render_template('calibrate_splash_instructions.html')


@app.route('/main_splash_instructions')
@login_required
def main_splash_instructions():
    """
    Displays the main splash page with navigation options to different sections of the application.
    """
    return render_template('main_splash_instructions.html')


@app.route('/splash_utilities_instructions')
@login_required
def splash_utilities_instructions():
    """
    Provides instructions for the utilities splash page, describing the tools available under utilities.
    """
    return render_template('splash_utilities_instructions.html')


@app.route('/masking_instructions')
@login_required
def masking_instructions():
    """
    Provides detailed instructions for masking, including digitizing and generating masks for PIV calculations.
    """
    return render_template('masking_instructions.html')


@app.route('/piv_parameters_instructions')
@login_required
def piv_parameters_instructions():
    """
    Provides instructions for configuring PIV parameters, explaining each parameter's role in the PIV process.
    """
    return render_template('piv_parameters_instructions.html')


@app.route('/trapezoid_instructions')
@login_required
def trapezoid_instructions():
    """
    Provides instructions for defining the trapezoid area used in PIV calculations.
    """
    return render_template('trapezoid_instructions.html')


@app.route('/setup_calibration_instructions')
@login_required
def setup_calibration_instructions():
    """
    Provides an overview of the setup and calibration process, guiding users through the necessary steps.
    """
    return render_template('setup_calibration_instructions.html')


@app.route('/piv_functions_instructions')
@login_required
def piv_functions_instructions():
    """
    Provides instructions for using PIV functions, including running tests, performing continuous PIV, and viewing results.
    """
    return render_template('piv_functions_instructions.html')


@app.route('/disk_space', methods=['GET'])
@login_required
def get_disk_space():
    """
    Retrieve disk current avalable disk space and calculates how long until it runs out
    """
    total, used, free = shutil.disk_usage("/")
    free_gb = free / (1024**3)  # Convert to GB

    with open(SAVE_CONFIG, 'r') as sc:
        save_conf = json.load(sc)
    current_dir = save_conf['current_data_directory']
    if os.path.exists(current_dir):
        mp4_files = [f for f in os.listdir(current_dir) if f.endswith('.mp4')]
        run_out_date_str = None
    else:
        run_out_date_str = "Cannot Calculate no recent data runs"
        mp4_files = None
    file_size = 0
    print(current_dir)
    if mp4_files:
        for dirpath, dirnames, filenames in os.walk(current_dir):
            for f in filenames:
                file_size += os.path.getsize(os.path.join(dirpath, f))
    else:
        if not run_out_date_str:
            run_out_date_str = "PIV running cannot calculate size of files."
    print(file_size)

    with open(MAIN_CONFIG, 'r') as cf:
        config = json.load(cf)
    try:
        length = int(config.get("site_piv_break"))
    except:
        length = 15
    if file_size != 0:

        run_duration_hours = (free // file_size) * length / 60
        run_out_date = datetime.now() + timedelta(hours=run_duration_hours)
        run_out_date_str = run_out_date.strftime("%m-%d-%y")

    return jsonify({
        "free_space_gb": round(free_gb, 2),
        "days_till_run_out": run_out_date_str
    })


@app.route('/stream_saving_to_usb')
def stream_saving_to_usb():
    '''
    Stream log file updates to the client.
    '''
    def generate():
        time.sleep(0.5)
        test_LOG = f'{BASE_DIR}/saving_to_usb.log'
        with open(test_LOG, "r") as log_file:
            log_file.seek(0, 2) 
            while True:
                line = log_file.readline()
                if line:
                    yield f"data: {line.strip()}\n\n"
                time.sleep(0.1) 

    return Response(stream_with_context(generate()), content_type='text/event-stream') 


@app.route('/usb_saving')
@login_required
def usb_saving():
    '''
    Redirects users to the waiting html
    '''
    return render_template('waiting_save_to_usb.html')

@app.route('/unmount_USB', methods=['GET'])
def unmount_USB():
    """
    Unmounts USB after saving
    """
    mount_point = find_usb_device()
    print ("mount point =", mount_point)
    if not mount_point:
        flash("No USB drive found or already unmounted.", "usb")
        return redirect(request.referrer or url_for('save_data'))

    try:

        subprocess.run(['fuser', '-k', mount_point], capture_output=True, text=True)

        result = subprocess.run(['umount', '-l', mount_point], capture_output=True, text=True)

        if result.returncode == 0:
            flash(f"USB drive unmounted successfully from {mount_point}!", "usb")
        else:
            flash(f"Error unmounting USB: {result.stderr}", "usb")


        subprocess.run(['eject', mount_point], capture_output=True, text=True)
    
    except Exception as e:
        flash(f"Exception occurred: {str(e)}", "usb")
    
  
    return redirect(request.referrer or url_for('splash'))  

# --- Goodness-based auto loop control ---

START_AT = 80   # start capturing when score >= this
STOP_AT  = 78   # stop capturing when score < this

PIV_PROC = None
IS_CAPTURING = False

def start_auto():
    global PIV_PROC, IS_CAPTURING, video_handler
    # stop preview so GStreamer can claim the device
    try:
        if video_handler:
            video_handler.stop()
    except Exception:
        pass

    # tell the shell loop to run and spawn it if not running
    (Path(BASE_DIR) / 'monitor_file.txt').write_text('run\n')
    if not PIV_PROC or (PIV_PROC.poll() is not None):
        PIV_PROC = subprocess.Popen(['bash', 'run_PIV.sh'], cwd=str(BASE_DIR))
    IS_CAPTURING = True

def stop_auto():
    global IS_CAPTURING
    (Path(BASE_DIR) / 'monitor_file.txt').write_text('stop\n')
    IS_CAPTURING = False
    # restart preview without blocking the request
    threading.Thread(target=_bring_preview_back, daemon=True).start()

def _bring_preview_back():
    global video_handler
    try:
        video_handler = VideoStreamHandler()  # this will wait until camera is free
    except Exception:
        pass


@app.route('/api/goodness', methods=['POST'])
def api_goodness():
    data = request.get_json(silent=True) or {}
    try:
        score = float(data.get('score'))
    except (TypeError, ValueError):
        return jsonify({'error': 'score must be a number 0-100'}), 400

    global IS_CAPTURING
    # hysteresis: start and stop thresholds are different
    if score >= START_AT and not IS_CAPTURING:
        start_auto()
    elif score < STOP_AT and IS_CAPTURING:
        stop_auto()

    return jsonify({
        'ok': True,
        'score': score,
        'capturing': IS_CAPTURING,
        'start_at': START_AT,
        'stop_at': STOP_AT
    })


if __name__ == "__main__":
    #start imu thread then the app
    start_imu_thread()
    app.run(host='0.0.0.0', threaded=True)
