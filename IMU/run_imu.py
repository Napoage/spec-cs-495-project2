import time
import smbus
import datetime
import json
import os
from imusensor.MPU9250 import MPU9250
"""
### Script Description
This Python script reads data from an MPU9250 IMU sensor, computes orientation angles, 
and logs the sensor data along with timestamps to a text file.

#### Key Functionalities:
1. **IMU Initialization and Calibration**:
   - Initializes the MPU9250 IMU sensor.
   - Loads calibration data from a JSON file for accurate sensor readings.

2. **Timestamp Generation**:
   - Retrieves the current timestamp in the Eastern time zone.
   
3. **Data Logging**:
   - Logs accelerometer readings (x, y, z) and Euler angles (roll, pitch, yaw) with timestamps.
   
4. **Main Loop**:
   - Reads sensor data and logs it to a text file every second.
   
 Credits: 
	-Funding for this project is provided through the USGS Next Generation Water Observing System (NGWOS) Research and Development program.
	-Engineered by: Deep Analytics LLC http://www.deepvt.com/
	-Authors: Makayla Hayes, Jaylene Naylor
"""

# Define paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)

# IMU Sensor Configuration
ADDRESS = 0x68
BUS = smbus.SMBus(1)
imu = MPU9250.MPU9250(BUS, ADDRESS)
imu.begin()
imu.loadCalibDataFromFile(os.path.join(BASE_DIR, "IMU/calib.json"))

# Load Save Configuration
with open(os.path.join(BASE_DIR, 'save.json'), 'r') as f:
    save_config = json.load(f)
SAVE_CONFIG_DIRECTORY = save_config.get('config_folder')
OUTPUT_FILE = os.path.join(SAVE_CONFIG_DIRECTORY, "imu_data.txt")


# Function to get current timestamp
def get_timestamp():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# Function to write data to file
def write_to_file(data, output_file):
    with open(output_file, "a") as file:
        file.write(data)


# Main loop
if __name__ == "__main__":
    while True:
        imu.readSensor()
        imu.computeOrientation()

        # Get sensor values
        accel_x, accel_y, accel_z = imu.AccelVals
        roll, pitch, yaw = imu.roll, imu.pitch, imu.yaw
        timestamp = get_timestamp()

        # Format data string
        data = (
            f"{timestamp} - Accel x: {accel_x:.2f} ; Accel y: {accel_y:.2f} ; Accel z: {accel_z:.2f}\n"
            f"{timestamp} - Roll: {roll:.2f} ; Pitch: {pitch:.2f} ; Yaw: {yaw:.2f}\n\n"
        )

        # Write data to file
        write_to_file(data, OUTPUT_FILE)
        time.sleep(1)
