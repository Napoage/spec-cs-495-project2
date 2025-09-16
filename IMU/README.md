# IMU Directory
This directory contains the necessary scripts and files for collecting and calibrating IMU (Inertial Measurement Unit) data for use in Particle Image Velocimetry (PIV) runs. It includes scripts for IMU calibration, testing, and data collection, along with the calibration file (calib.json) that holds the IMU calibration parameters.

## Directory Structure
- `misc/`: Contains additional scripts for IMU-related tasks.
    - `calibrate_imu.py`: A script for calibrating the IMU.
    - `find.py`: A utility script for finding and locating the IMU devices.
    - `testimu.py`: A script for testing the IMU functionality and ensuring that it is working correctly.

- `calib.json`: The calibration file that contains the IMU calibration parameters. This file is used to apply the proper transformations and corrections to IMU data.

- `run_imu.py`: The main script that is called to collect IMU data for PIV runs. It captures IMU readings and stores the data for further processing in PIV analysis.

## Calibration File
The calib.json file contains the IMU calibration data and is used during PIV analysis to apply any necessary transformations or corrections to the raw IMU data. Ensure that the file is present and up-to-date before running any PIV processing.