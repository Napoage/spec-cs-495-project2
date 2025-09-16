# PIV Directory

This directory contains all the necessary scripts and functions for performing Particle Image Velocimetry (PIV) calculations. It houses the core functions, helper scripts, and image preprocessing steps required to execute and process PIV computations on image data.

## Core Functions

- `call_PIV_lab`: The main function for running the PIV calculations. It reads image paths and passes them along with configuration variables to the ensemble_PIV function.
- `ensembleP_PIV`: Processes batches of image pairs for PIV analysis and aggregates the results for further analysis and post-processing.

## Helper Scripts
These scripts assist with the PIV calculations and other preprocessing steps:

- `dctnmat`: Performs discrete cosine transform-related operations.
- `idctnmat`: Performs inverse discrete cosine transform operations.
- `inpaint_nans`: Fills in missing values (NaNs) in image data.
- `madilt2`: Applies median absolute deviation filter for image analysis.
- `post_process`: Handles the post-processing of results obtained from PIV calculations.
- `smoothn`: Applies smoothing functions to the processed images and results.

## Preprocessing

- `preprocess_frames`: This script is responsible for preparing raw image frames for PIV analysis. The raw frames are undistorted, homography transformations are applied, masking and enhancement processes are completed, and the frames are saved as images ready for the next steps.

## Process Flow
- `Raw Frames`: Frames are read from the raw_frames directory after being collected. These frames undergo undistortion, homography corrections,     masking, and enhancement. Once processed, the frames are saved as images.

- `call_PIV_lab`: This function reads the processed file paths and sends them to the ensemble_PIV function, along with the configuration variables.

- `ensemble_PIV`: The image files are batched, and each batch of image pairs is processed. Results from all batches are combined during post-processing.

- `Post-processing`: After PIV calculations are completed, the results are aggregated and saved as CSV files.

## Output
The results of the PIV calculations are saved as CSV files in the output directory for visualization.