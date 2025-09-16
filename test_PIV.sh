#!/bin/bash
#
# Script Name: test_PIV.sh
# Description: This script automates the process of capturing, preprocessing, 
#              and analyzing image frames for PIV (Particle Image Velocimetry).
#              It handles IMU data collection, frame extraction, preprocessing,
#              and calls the PIV analysis script.
#
# Usage:
#   Script is run by web-app when user selects Run Test
#
# Credits:
#	-Funding for this project is provided through the USGS Next Generation Water Observing System (NGWOS) Research and Development program.
#	-Engineered by: Deep Analytics LLC http://www.deepvt.com/
# -----------------------------------------------------


# TODO: this path should be discovered instead of hard-coded
PARENT_DIR='/home/spec/spec'
IMU_script="sudo python3 ${PARENT_DIR}/IMU/run_imu.py --unique-tag=IMUProcess"
LOG_FILE="${PARENT_DIR}/script.log"

# load gstreamer run_gst_launch() function
source ${PARENT_DIR}/common_functions.sh

#used to log messages to webpage
log_message() {
    echo "$1" >> "$LOG_FILE"
}

# Log the start of the script
echo "Script started at $(date)"
log_message "Test PIV process started"

mkdir -p "${PARENT_DIR}/raw_frames"

# Path to your config.json file
CONFIG_FILE="${PARENT_DIR}/config.json"

# Parse frameInterval and duration using jq
frame_interval=$(jq -r '.frameInterval' "$CONFIG_FILE")
duration=$(jq -r '.capture_time' "$CONFIG_FILE")
width=$(jq -r '.reduced_image_width' "$CONFIG_FILE")
height=$(jq -r '.reduced_image_height' "$CONFIG_FILE")

# Check if jq parsing was successful
if [ -z "$frame_interval" ] || [ -z "$duration" ]; then
    echo "Error: Could not retrieve frameInterval or duration from $CONFIG_FILE."
    log_message "Error: Could not retrieve frameInterval or duration from $CONFIG_FILE."
    exit 1
fi

# Convert frameInterval to a framerate value
framerate=$(printf "%.0f" $(bc -l <<< "1/$frame_interval"))

# Step 1: start preprocesssing
echo "Running IMU command"
# log_message "Running IMU command"
$IMU_script &
IMU_PID=$!

# clearing any existing raw_frames
echo "Clearing old frames..."
rm -f ${PARENT_DIR}/raw_frames/*

# Run gst-launch with infinite retry mechanism
log_message "Collecting images at framerate ${framerate} frames a second for ${duration} seconds..."
if ! run_gst_launch; then
    echo "Unexpected error in run_gst_launch function"
    cleanup
    continue  # Continue the main loop instead of exiting
fi

# safety check to make sure raw_frames is populated
if [ ! $(ls -al ${PARENT_DIR}/raw_frames | wc -l) -ge $duration ]; then
    echo "Error, no raw frames detected! Retrying..."
    cleanup
    continue  # Continue the main loop instead of exiting
else
    echo "raw frames detected, proceeding."
fi


#preprocess images
log_message "Preprocessing images"
python3 ${PARENT_DIR}/PIV/preprocess_frames.py
log_message "Finished preprocssing images"
cleanup


# callPIVLAP.py and wait for it to finish
echo "Running callPIVLAP.py..."
log_message "Running call_PIV_lab.py..."
pushd $PARENT_DIR
python3 ${PARENT_DIR}/PIV/call_PIV_lab.py
popd

#delete frames
echo "Deleting all images in ./images/ folder..."
rm -f ${PARENT_DIR}/images/*
rm -f ${PARENT_DIR}/raw_frames/*

# Log the end of the script
echo "Script completed at $(date)"
log_message "Test completed"
> "$LOG_FILE"
