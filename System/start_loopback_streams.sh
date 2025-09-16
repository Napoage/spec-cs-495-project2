#!/bin/bash
#
# Script Name: start_loopback_streams.sh
# Description: Starts gstreamer loopback streams
#
# Usage:
#   Called by systemd rule
#
# Credits:
#	-Funding for this project is provided through the USGS Next Generation Water Observing System (NGWOS) Research and Development program.
#	-Engineered by: Deep Analytics LLC http://www.deepvt.com/
# -----------------------------------------------------

echo "Starting GStreamer"

# This command starts streaming from the main camera (/dev/video0)
# and copies the stream to two virtual cameras (/dev/video2, /dev/video3)
gst-launch-1.0 -e v4l2src device=/dev/video0 ! image/jpeg,width=1920,height=1080,framerate=30/1 ! jpegdec ! videoconvert ! tee name=t t. \
     ! queue ! v4l2sink device=/dev/video2 t. \
     ! queue ! v4l2sink device=/dev/video3