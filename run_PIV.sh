#!/bin/bash
#
# Script Name: run_PIV.sh
# Description: This script automates the process of capturing, preprocessing, 
#              and analyzing image frames for PIV (Particle Image Velocimetry).
#              It handles IMU data collection, frame extraction, preprocessing,
#              and calls the PIV analysis script once, then exits.
#
# Usage:
#   Script is run by confidence_loop() in app.py
#
# Credits:
#	-Funding for this project is provided through the USGS Next Generation Water Observing System (NGWOS) Research and Development program.
#	-Engineered by: Deep Analytics LLC http://www.deepvt.com/
# -----------------------------------------------------


# Monitor file to watch
#PARENT_DIR='/home/spec/spec'
PARENT_DIR="$(cd "$(dirname "$0")" && pwd)"
monitor_file="${PARENT_DIR}/monitor_file.txt"
IMU_script="sudo python3 ${PARENT_DIR}/IMU/run_imu.py --unique-tag=IMUProcess"
# Path to your config.json file
CONFIG_FILE="${PARENT_DIR}/config.json"
LOG_FILE="${PARENT_DIR}/script.log"
VIDEO_PATH="${PARENT_DIR}/mock_hardware/Water_Moving.mp4"
# --- logging & safety (add these) ---
touch "$LOG_FILE"                          # clear log each fresh start
exec > >(tee -a "$LOG_FILE") 2>&1          # send stdout/stderr to script.log
set -Eeuo pipefail                         # fail fast on errors
echo "$(date -Is) PIV SCRIPT STARTED"


mkdir -p "${PARENT_DIR}/raw_frames"

# load gstreamer run_gst_launch() function
source $PARENT_DIR/mock_hardware/common_functions.sh

trap cleanup EXIT INT TERM

capture_frames_from_video() {
  local video="$1"
  local framerate="$2"   # frames per second (can be decimal)
  local width="$3"
  local height="$4"

  echo "Extracting frames from video: $video"
  rm -f "${PARENT_DIR}/raw_frames/"* 2>/dev/null || true

  # ffmpeg: sample at the desired FPS and resize to your reduced resolution
  ffmpeg -hide_banner -loglevel error -y \
    -i "$video" \
    -vf "fps=${framerate},scale=${width}:${height}:flags=lanczos" \
    "${PARENT_DIR}/raw_frames/%06d.jpg"
}

# Read site_piv_break (kept for potential future use)
site_piv_break=$(jq -r '.site_piv_break' "$CONFIG_FILE")

# Validate site_piv_break
if [[ ! "$site_piv_break" =~ ^[0-9]+$ ]]; then
  echo "Invalid site_piv_break value. Defaulting to 1 minute."
  site_piv_break=1
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
# Run capture (camera OR video) with infinite retry behavior similar to before
if [[ -n "$VIDEO_PATH" && -f "$VIDEO_PATH" ]]; then
  # Use the same values you read from config.json
  framerate=$(printf "%.6f" $(bc -l <<< "1/$frame_interval"))
  capture_frames_from_video "$VIDEO_PATH" "$framerate" "$width" "$height" || {
    echo "Unexpected error extracting frames from video"
    cleanup
    exit 1
  }
else
  # Original camera path
  if ! capture_frames; then
    echo "Unexpected error in capture_frames function"
    cleanup
    exit 1
  fi
fi

# safety check to make sure raw_frames is populated
if [ ! $(ls -al ${PARENT_DIR}/raw_frames | wc -l) -ge $duration ]; then
  echo "Error, no raw frames detected!"
  cleanup
  exit 1
else
  echo "raw frames detected, proceeding."
fi

cleanup

export MPLBACKEND=Agg
# Process images
python3 ${PARENT_DIR}/PIV/preprocess_frames.py
python3 ${PARENT_DIR}/PIV/call_PIV_lab.py
python3 ${PARENT_DIR}/mock_hardware/visualize_csv_data.py || echo "visualize_csv_data failed"
rm -f ${PARENT_DIR}/images/*
rm -f ${PARENT_DIR}/raw_frames/*

echo "One-shot PIV run complete at $(date -Is)"
exit 0