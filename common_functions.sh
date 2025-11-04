#!/bin/bash
#
# Script Name: common_functions.sh
# Description: This scripts has the common functions Gstreamer and cleanup for run_PIV.sh and test_PIV.sh.
#
# Usage:
#   Script is runs when called by run_PIV.sh or test_PIV.sh
#
# Credits:
#	-Funding for this project is provided through the USGS Next Generation Water Observing System (NGWOS) Research and Development program.
#	-Engineered by: Deep Analytics LLC http://www.deepvt.com/
# -----------------------------------------------------

# Function to clean up the IMU process
cleanup() {
    echo "Stopping IMU process..."
    # Kill the IMU process
    sudo pkill -f "run_imu.py"
    echo "IMU process stopped."
}

# Launch gstreamer for a finite time based on duration parameter, retrying if needed
run_gst_launch() {
    local attempt=1
    local success=false
    local expected_frames=$((duration * framerate))
    local pid
    local frame_count
    local final_count
    local tolerance=8  # Define tolerance for frame count capture

    while [ "$success" = false ]; do
        echo "Attempt $attempt to capture $expected_frames frames (±$tolerance frames acceptable)..."

        # Clear any existing frames before starting
        rm -f ${PARENT_DIR}/raw_frames/*
        # Clear any existing video output
        rm -f "${PARENT_DIR}/save_data/video_output.mp4"

        # Start gst-launch in background
        gst-launch-1.0 -e -m -f v4l2src device=/dev/video3 ! \
            video/x-raw,format=YUY2,width=${width},height=${height},framerate=30/1 ! \
            tee name=t \
            t. ! queue max-size-buffers=2 leaky=downstream ! videoconvert ! \
                x264enc bitrate=5000 speed-preset=ultrafast tune=zerolatency ! \
                mp4mux streamable=true fragment-duration=1 ! \
                filesink location=${PARENT_DIR}/save_data/video_output.mp4 sync=false \
            t. ! queue max-size-buffers=2 leaky=downstream ! videoconvert ! videorate ! \
                video/x-raw,framerate=${framerate}/1 ! jpegenc ! \
                multifilesink async=false sync=false max-lateness=0 qos=false location=${PARENT_DIR}/raw_frames/frame%05d.jpg &

        pid=$!
        echo "gst-launch started with PID: $pid"

        # Monitor frame count
        local pipeline_failed=false
        while true; do
            # Check if gst-launch is still running
            if ! kill -0 $pid 2>/dev/null; then
                echo "gst-launch terminated unexpectedly"
                pipeline_failed=true
                break
            fi

            # Count current frames
            frame_count=$(ls ${PARENT_DIR}/raw_frames/*.jpg 2>/dev/null | wc -l)
            echo -ne "Current frames: $frame_count/$expected_frames\r"

            if [ "$frame_count" -ge "$expected_frames" ]; then
                echo -e "\nDesired frame count reached"

                # Brief moment for final writes
                sleep 0.1

                # Terminate gst-launch gracefully first
                kill -SIGINT $pid
                sleep 2

                # If still running, force terminate
                if kill -0 $pid 2>/dev/null; then
                    echo "Forcing termination of gst-launch"
                    kill -9 $pid
                fi

                # Wait for any children to finish
                wait $pid 2>/dev/null
                break
            fi

            sleep 0.1
        done

        # Verify final frame count
        final_count=$(ls ${PARENT_DIR}/raw_frames/*.jpg 2>/dev/null | wc -l)
        echo "Final frame count: $final_count/$expected_frames"

        # Check if count is within tolerance
        if [ "$final_count" -ge "$((expected_frames - tolerance))" ] && \
           [ "$final_count" -le "$((expected_frames + tolerance))" ]; then
            echo "Frame capture completed successfully on attempt $attempt"
            echo "Got $final_count frames (within ±$tolerance of target $expected_frames)"

            # Verify MP4 file exists and has size greater than 0
            if [ -f "${PARENT_DIR}/save_data/video_output.mp4" ] && \
               [ -s "${PARENT_DIR}/save_data/video_output.mp4" ]; then

                # If we have extra frames, remove them
                if [ "$final_count" -gt "$expected_frames" ]; then
                    echo "Trimming excess frames..."
                    ls -v ${PARENT_DIR}/raw_frames/*.jpg | \
                    tail -n $((final_count - expected_frames)) | \
                    xargs rm -f

                    # Verify the trim operation
                    final_count=$(ls ${PARENT_DIR}/raw_frames/*.jpg 2>/dev/null | wc -l)
                    echo "After trimming: $final_count frames"

                    if [ "$final_count" -eq "$expected_frames" ]; then
                        echo "Successfully trimmed to exact frame count"
                        success=true
                    else
                        echo "Error in trimming process, retrying capture..."
                        ((attempt++))
                        continue
                    fi
                else
                    success=true
                fi
            else
                echo "MP4 file is missing or empty, retrying..."
                ((attempt++))
            fi
        else
            echo "Frame count outside acceptable range on attempt $attempt"
            echo "Got $final_count frames, expected $expected_frames (±$tolerance)"
            echo "Waiting 5 seconds before retry..."
            sleep 5
            ((attempt++))

            if [ $attempt -gt 10 ]; then
                echo "Maximum retry attempts reached. Please check the system."
                return 1
            fi
        fi
    done

    # Final verification
    final_count=$(ls ${PARENT_DIR}/raw_frames/*.jpg 2>/dev/null | wc -l)
    echo "Final verification: $final_count frames"
    if [ "$final_count" -eq "$expected_frames" ]; then
        echo "Frame capture completed successfully"
        return 0
    else
        echo "Unexpected final frame count: $final_count"
        return 1
    fi
}

check_readable() {
    if [ ! -r "$1" ]; then
        echo "Error: Cannot read $1"
        return 1
    fi
    return 0
}

is_numeric() {
    [[ "$1" =~ ^[0-9]+$ ]]
}

enable_rtc_charging() {
    local BATTERY_VOLTAGE_PATH="/sys/devices/platform/soc/soc:rpi_rtc/rtc/rtc0/battery_voltage"
    local CHARGING_VOLTAGE_PATH="/sys/devices/platform/soc/soc:rpi_rtc/rtc/rtc0/charging_voltage"
    local CONFIG_PATH="/boot/firmware/config.txt"
    local CHARGING_PARAM="dtparam=rtc_bbat_vchg=3000000"

    # Check if files exist and are readable
    for file in "$BATTERY_VOLTAGE_PATH" "$CHARGING_VOLTAGE_PATH" "$CONFIG_PATH"; do
        if ! check_readable "$file"; then
            return 1
        fi
    done

    # Read battery and charging voltages
    local battery_voltage=$(cat "$BATTERY_VOLTAGE_PATH")
    local charging_voltage=$(cat "$CHARGING_VOLTAGE_PATH")

    # Check if values are numeric
    if ! is_numeric "$battery_voltage" || ! is_numeric "$charging_voltage"; then
        echo "Error: Invalid voltage values"
        return 1
    fi

    echo "Battery voltage: $battery_voltage"
    echo "Charging voltage: $charging_voltage"

    # Check if battery is present (voltage > 0) and charging is not enabled
    if [ "$battery_voltage" -gt 0 ] && [ "$charging_voltage" -eq 0 ]; then
        echo "Battery detected and charging is not enabled"

        # Check if charging parameter already exists in config.txt
        if ! grep -q "rtc_bbat_vchg" "$CONFIG_PATH"; then
            echo "Adding charging parameter to config.txt"

            # Backup config file
            sudo cp "$CONFIG_PATH" "${CONFIG_PATH}.backup"

            # Add charging parameter
            echo "$CHARGING_PARAM" | sudo tee -a "$CONFIG_PATH"

            echo "Charging parameter added. A reboot is required for changes to take effect."
            return 2  # Return code 2 indicates reboot needed
        else
            echo "Charging parameter already exists in config.txt"
            return 0
        fi
    else
        if [ "$battery_voltage" -eq 0 ]; then
            echo "No battery detected"
        fi
        if [ "$charging_voltage" -gt 0 ]; then
            echo "Charging is already enabled"
        fi
        return 0
    fi
}