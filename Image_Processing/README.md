# Image Processing Directory
This directory contains scripts and data necessary for camera calibration and image processing tasks.

## Directory Structure
- `misc/`: Contains additional scripts related to image processing.
    - `calibration.py`: A script for performing camera calibration.
    - `take_photos.py`: A script for taking photos to be used in camera calibration.
    - /images_for_calibration: Stores the images the user takes with the take_photos.py script. These will be used by the calibration.py script.
- `camera calibration matrices`: Files containing camera calibration data.
- `distortion coefficients`: Parameters that describe the distortion introduced by the camera lens.

## Camera Calibration
The directory holds the necessary calibration data, including:
- **Camera Calibration Matrices**: Used for correcting lens distortions.
- **Distortion Coefficients**: Parameters that describe the distortion introduced by the camera lens.

Ensure that these files are properly generated and stored before using them in image processing tasks.
