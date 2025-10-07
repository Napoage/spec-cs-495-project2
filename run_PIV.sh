#!/bin/bash
#
# Script Name: run_PIV.sh
# Description: Automates capturing, preprocessing, and analyzing image frames for PIV.
#              For demo mode, you can disable IMU, camera funcs, and processing via toggles below.
#
# Usage:
#   Script is run by web-app when user selects Run PIV
#
# Credits:
#   - Funding: USGS NGWOS R&D
#   - Engineered by: Deep Analytics LLC http://www.deepvt.com/
# -----------------------------------------------------

# -------- demo toggles (flip 1->0 to enable) --------
DISABLE_IMU=1
DISABLE_CAMERA_FUNCS=1      # don't source common_functions.sh or call capture_frames
SKIP_PREPROCESS=1
SKIP_PIV=0
SKIP_VIS=0                  # set to 0 only if you have CSVs ready to visualize
# ----------------------------------------------------

# Base paths
PARENT_DIR="$(cd "$(dirname "$0")" && pwd)"

# -------- single-instance guard (global) --------
LOCKFILE="/var/lock/spec_piv_runner.lock"
PIDFILE="${PARENT_DIR}/.run_piv.pid"
mkdir -p /var/lock || true

# lock across the whole system (same file regardless of cwd)
exec 9> "$LOCKFILE"
if ! flock -n 9; then
  echo "Another run_PIV.sh holds the global lock; exiting."
  exit 0
fi

# PID file check (extra safety if someone reused the lock incorrectly)
if [ -f "$PIDFILE" ]; then
  oldpid="$(cat "$PIDFILE" 2>/dev/null || true)"
  if [ -n "$oldpid" ] && kill -0 "$oldpid" 2>/dev/null; then
    echo "run_PIV.sh already running as PID $oldpid; exiting."
    exit 0
  fi
fi
echo $$ > "$PIDFILE"
cleanup_pid() { rm -f "$PIDFILE"; }
trap cleanup_pid EXIT
# -----------------------------------------------


monitor_file="${PARENT_DIR}/monitor_file.txt"

# Configs & logs
CONFIG_FILE="${PARENT_DIR}/config.json"
LOG_FILE="${PARENT_DIR}/script.log"

# Video source (used when no camera / for demos)
VIDEO_PATH="${PARENT_DIR}/Water_Moving.mp4"

# Optional: sudo password file helper (only needed if you actually run sudo commands)
sudo_pwfile="${HOME}/.sudo_cred"
run_sudo() {
  if [ ! -f "$sudo_pwfile" ]; then
    echo "Missing sudo password file: $sudo_pwfile" >&2
    return 1
  fi
  cat "$sudo_pwfile" | sudo -S -p '' "$@"
}

# --- logging & safety ---
touch "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1
set -Eeuo pipefail
echo "$(date -Is) PIV SCRIPT STARTED"

mkdir -p "${PARENT_DIR}/raw_frames"

# Load camera helpers only if enabled
if [ "${DISABLE_CAMERA_FUNCS}" -eq 0 ]; then
  # provides capture_frames() and cleanup()
  # shellcheck disable=SC1090
  source "$PARENT_DIR/common_functions.sh"
fi

# Ensure cleanup calls are safe even if not sourced
cleanup_run() {
  if declare -F cleanup >/dev/null; then
    cleanup || true
  fi
  if [[ "${IMU_PID:-}" =~ ^[0-9]+$ ]]; then
    kill "${IMU_PID}" 2>/dev/null || true
  fi
}
trap cleanup_run EXIT INT TERM

capture_frames_from_video() {
  local video="$1"
  local framerate="$2"   # frames per second (can be decimal)
  local width="$3"
  local height="$4"

  echo "Extracting frames from video: $video"
  rm -f "${PARENT_DIR}/raw_frames/"* 2>/dev/null || true

  ffmpeg -hide_banner -loglevel error -y \
    -i "$video" \
    -vf "fps=${framerate},scale=${width}:${height}:flags=lanczos" \
    "${PARENT_DIR}/raw_frames/%06d.jpg"
}

# --------------- main loop ---------------
while true; do
  if [ -f "$monitor_file" ]; then
    file_content=$(cat "$monitor_file")

    if [ "$file_content" == "run" ]; then
      # Read schedule/params from config
      site_piv_break=$(jq -r '.site_piv_break' "$CONFIG_FILE" || echo "1")
      if [[ ! "$site_piv_break" =~ ^[0-9]+$ ]]; then
        echo "Invalid site_piv_break value. Defaulting to 1 minute."
        site_piv_break=1
      fi

      frame_interval=$(jq -r '.frameInterval' "$CONFIG_FILE")
      duration=$(jq -r '.capture_time' "$CONFIG_FILE")
      width=$(jq -r '.reduced_image_width' "$CONFIG_FILE")
      height=$(jq -r '.reduced_image_height' "$CONFIG_FILE")

      if [ -z "$frame_interval" ] || [ -z "$duration" ] || [ -z "$width" ] || [ -z "$height" ]; then
        echo "Error: Could not retrieve frameInterval/duration/width/height from config."
        exit 1
      fi

      # Start IMU if enabled
      if [ "${DISABLE_IMU}" -eq 0 ]; then
        echo "Running IMU command"
        # If IMU needs root: use run_sudo; otherwise plain python3.
        run_sudo python3 "${PARENT_DIR}/IMU/run_imu.py" --unique-tag=IMUProcess &
        IMU_PID=$!
      else
        echo "IMU disabled for demo"
      fi

      # Calculate desired FPS from frame_interval
      framerate=$(printf "%.6f" "$(bc -l <<< "1/$frame_interval")")
      echo "Starting process with framerate ~${framerate} fps for ${duration}s..."

      echo "Clearing old frames..."
      rm -f "${PARENT_DIR}/raw_frames/"* 2>/dev/null || true

      # Capture frames (video file preferred for demo)
      if [[ -n "$VIDEO_PATH" && -f "$VIDEO_PATH" ]]; then
        capture_frames_from_video "$VIDEO_PATH" "$framerate" "$width" "$height" || {
          echo "Unexpected error extracting frames from video"
          if declare -F cleanup >/dev/null; then cleanup || true; fi
          continue
        }
      else
        if [ "${DISABLE_CAMERA_FUNCS}" -eq 0 ]; then
          if ! capture_frames; then
            echo "Unexpected error in capture_frames function"
            if declare -F cleanup >/dev/null; then cleanup || true; fi
            continue
          fi
        else
          echo "Camera capture disabled and no VIDEO_PATH present; waiting…"
          sleep 5
          continue
        fi
      fi

      # Safety check for frames
      if [ ! "$(ls -1 "${PARENT_DIR}/raw_frames" 2>/dev/null | wc -l)" -ge 1 ]; then
        echo "Error, no raw frames detected! Retrying..."
        if declare -F cleanup >/devnull; then cleanup || true; fi
        continue
      else
        echo "Raw frames detected, proceeding."
      fi

      # cleanup camera pipeline if present
      if declare -F cleanup >/dev/null; then cleanup || true; fi

      export MPLBACKEND=Agg

      # -------- processing stage toggles --------
      if [ "${SKIP_PREPROCESS}" -eq 0 ]; then
        python3 "${PARENT_DIR}/PIV/preprocess_frames.py"
      else
        echo "Skipping preprocess_frames for demo"
      fi

      if [ "${SKIP_PIV}" -eq 0 ]; then
        python3 "${PARENT_DIR}/PIV/call_PIV_lab.py"
      else
        echo "Skipping call_PIV_lab for demo"
      fi

      if [ "${SKIP_VIS}" -eq 0 ]; then
        python3 "${PARENT_DIR}/visualize_csv_data.py" || echo "visualize_csv_data failed"
      else
        echo "Skipping visualize step for demo"
      fi
      # -----------------------------------------

      # Tidy temp dirs (safe if empty)
      rm -f "${PARENT_DIR}/images/"* 2>/dev/null || true
      rm -f "${PARENT_DIR}/raw_frames/"* 2>/dev/null || true

      # Schedule next run on the minute grid defined by site_piv_break
      current_time=$(date +%s)
      next_run_time=$(( (current_time / (site_piv_break * 60) + 1) * (site_piv_break * 60) ))
      sleep_time=$(( next_run_time - current_time ))

      echo "Cycle complete at $(date -Is)"
      echo "Sleeping ${sleep_time}s (until $(date -d @$next_run_time -Is))"
      sleep "$sleep_time"

    elif [ "$file_content" == "stop" ]; then
      echo "Stopped. Checking again in 5 seconds..."
      sleep 5
    else
      echo "Unknown command in monitor file. Waiting..."
      sleep 5
    fi
  else
    echo "Monitor file not found. Waiting..."
    sleep 5
  fi
done
