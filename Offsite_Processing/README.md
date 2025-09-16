# Frame Processing Script
This script processes frames from a video for offsite analysis. Once processed, these frames can be run through **TRiVIA** for Particle Image Velocimetry (PIV) processing.

## Setup
1. Place your video file and configuration file in the same directory as this script.
2. Run the script to start the frame extraction and processing.

## Configuration
If you need a different frame rate, modify the frameInterval variable in the configuration file to the desired value.
**Note:** The video has a maximum framerate of 30 frames per second.

# Key Functionalities
1. **Frame Extraction**
    - Extracts frames from the video at the specified framerate.
2. **Undistortion and Homography Transformation**
    - Undistorts frames using camera calibration data.
    - Applies a perspective transformation (homography) to extract a trapezoidal region of the frame.
3. **Saving Processed Frames**
    - Saves the processed frames into a folder named homographyFrames.