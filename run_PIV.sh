#!/bin/bash
#
# Script Name: run_PIV.sh
# Description: This script automates the process of capturing, preprocessing, 
#              and analyzing image frames for PIV (Particle Image Velocimetry).
#              It handles IMU data collection, frame extraction, preprocessing,
#              and calls the PIV analysis script. This dcript will continuously run on
#              user defined intervals. 
#
# Usage:
#   Script is run by web-app when user selects Run PIV
#
# Credits:
#	-Funding for this project is provided through the USGS Next Generation Water Observing System (NGWOS) Research and Development program.
#	-Engineered by: Deep Analytics LLC http://www.deepvt.com/
# -----------------------------------------------------


# Monitor file to watch
PARENT_DIR='/home/spec/spec'
monitor_file="${PARENT_DIR}/monitor_file.txt"
IMU_script="sudo python3 ${PARENT_DIR}/IMU/run_imu.py --unique-tag=IMUProcess"
# Path to your config.json file
CONFIG_FILE="${PARENT_DIR}/config.json"
LOG_FILE="${PARENT_DIR}/script.log"
echo 'PIV SCRIPT STARTED'


mkdir -p "${PARENT_DIR}/raw_frames"

# load gstreamer run_gst_launch() function
source $PARENT_DIR/common_functions.sh

# Main loop to watch monitor_file.txt and run PIV calculations continuously
while true; do
  if [ -f "$monitor_file" ]; then
    file_content=$(cat "$monitor_file")

    if [ "$file_content" == "run" ]; then
      # Read site_piv_break on each run in case it's updated
      site_piv_break=$(jq -r '.site_piv_break' "$CONFIG_FILE")

      # Validate site_piv_break
      if [[ ! "$site_piv_break" =~ ^[0-9]+$ ]]; then
        echo "Invalid site_piv_break value. Defaulting to 15 minutes."
        site_piv_break=15
      fi

      frame_interval=$(jq -r '.frameInterval' "$CONFIG_FILE")
      duration=$(jq -r '.capture_time' "$CONFIG_FILE")
      width=$(jq -r '.reduced_image_width' "$CONFIG_FILE")
      height=$(jq -r '.reduced_image_height' "$CONFIG_FILE")

      if [ -z "$frame_interval" ] || [ -z "$duration" ]; then
        echo "Error: Could not retrieve frameInterval or duration."
        exit 1
      fi

      # Run IMU
      echo "Running IMU command"
      $IMU_script &
      IMU_PID=$!

      framerate=$(printf "%.0f" $(bc -l <<< "1/$frame_interval"))
      echo "Starting process with framerate ${framerate}/1 for ${duration} seconds..."

      # clearing any existing raw_frames
      echo "Clearing old frames..."
      rm -f ${PARENT_DIR}/raw_frames/*

      # Run gst-launch with infinite retry mechanism
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

      cleanup

      # Process images
      python3 ${PARENT_DIR}/PIV/preprocess_frames.py
      python3 ${PARENT_DIR}/PIV/call_PIV_lab.py
      rm -f ${PARENT_DIR}/images/*
      rm -f ${PARENT_DIR}/raw_frames/*
      > "$LOG_FILE"
      # Calculate next scheduled run
      current_time=$(date +%s)
      next_run_time=$(( (current_time / (site_piv_break * 60) + 1) * (site_piv_break * 60) ))
      sleep_time=$((next_run_time - current_time))

      echo "Sleeping until $(date -d @$next_run_time)..."
      sleep "$sleep_time"

    elif [ "$file_content" == "stop" ]; then
      echo "Stopped. Checking again in 5 seconds..."
      sleep 5
    else
      echo "Unknown command. Waiting..."
      sleep 5
    fi
  else
    echo "Monitor file not found. Waiting..."
    sleep 5
  fi
done
