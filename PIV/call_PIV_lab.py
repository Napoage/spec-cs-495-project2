import numpy as np
import sys

sys.path.append(".")
from ensemble_PIV import ensemble_piv
import numpy as np
import json
import cv2
import os
import re
import pandas as pd
import shutil
"""
 PIV Analysis Script for Velocity Field Computation

 This script processes a stack of images using the Particle Image Velocimetry (PIV) technique.
 It applies a polygon mask to restrict the area of interest, scales velocity vectors, 
 and outputs the results in CSV format. It also supports configurable parameters 
 for image processing, velocity filtering, and output formatting. The script includes 
 functions for loading configurations, preprocessing images, applying PIV, and saving results.

 Functions:
 - inpolygon(x, y, vertices): Check if points are inside the defined polygon.
 - call_pivlab(stack, piv_params, save_config, BASE_DIR): Main function for running the PIV algorithm.
 - load_config(save_config_path): Loads the configuration from a JSON file.
 - get_filepaths(image_directory, pattern): Loads and sorts image files from a directory based on a given pattern.
 
 Note:
 This script assumes a pre-processing step to ensure the images are correctly aligned and ready for PIV analysis.
 
 Credits: 
	-Funding for this project is provided through the USGS Next Generation Water Observing System (NGWOS) Research and Development program.
	-Engineered by: Deep Analytics LLC http://www.deepvt.com/
	-Ported from:
		 Legleiter, C.J., 2024, TRiVIA - Toolbox for River Velocimetry using Images from Aircraft (ver. 2.1.3, September, 2024): 
				 U.S. Geological software release, https://doi.org/10.5066/P9AD3VT3.
	-Authors: Makayla Hayes

"""


def inpolygon(x, y, vertices):
    """
    Helper function to check if points are within the set polygon
    
    Args:
        -x: array
            vector array of x velocity coordinates
        -y: array
            vector array of y velocity coordinates
        -vertices: array
            array of polygon dimensions
    Returns:
        logical array corresponding to the points that are in the polygon True=in False=out
        
    """
    # Check if each coordinate is within the desired area
    valid_coords = np.ones_like(x, dtype=bool)

    for i in range(len(x)):
        x_p = x[i]
        y_p = y[i]
        x_p = int(x_p)
        y_p = int(y_p)
        if vertices[y_p, x_p] == 0:
            valid_coords[i] = False

    return valid_coords


def call_pivlab(stack, piv_params, save_config, BASE_DIR):
    """
    This function is called to help set up for the main ensemblePIV algorithm
    
    Args:
        -stack: dictionary
            Input dictionary with variables 
                pre_proc - List of image paths
                final_roi - A mask delineating the ROI for the PIV analysis
                Rcrop - []
        -pixSize: float
            Size of pixels of the input images in meters representing the ground sampling distance GSD
        -frameInterval: float
            original capture interval for the images, in seconds
        -piv_params: dictionary 
            -img_field: String with name of the structured array containing image paths
            -pixSize: Pixel size of image sequence in meters
            -frameInterval: Original capture interval for the images, in seconds
            -passes: Scalar specifying the number of passes of the core PIV algorithm to complete hard set to 1 for the moment
            -intAreas: Vector with passes entries specifying the size of the interrogation area at each pass
            -minvel: Minimum velocity threshold used to filter out suspiciously low velocities. No filter is applied if empty or negative
            -maxvel: Maximum velocity threshold used to filter out suspiciously high velocities. No filter is applies if empty or negative
            -stdThresh: Threshold number of standard deviations, calculate at the reach scale, beyond which a velocity is an outlier and filtered out. 
                        If an empty array or negative number is passed no filter is applied
            -medianFilt: Threshold value of the difference between a velocity estimate and the local median beyond which
                the estimate will be considered an outlier and filtered out. If an empty array or negative number is passed no filter is applied
            -infillFlag: Logical flag to infill missing values in initial PIV output from each pass. Input should be True or 1
                to preform infilling, or if an empty array, 0, or negative number is passes no infilling is applied
            -smoothFlag: Logical flag to smooth missing values in initial PIV output from each pass. Input should be True or 1
                to preform infilling, or if an empty array, 0, or negative number is passes no infilling is applied
    
    Saves:
        -PIV Output CSV files:
            -xPiv: Array of PIV vector origin image column (x) coordinates (pixels)
            -yPiv: Array of PIV vector origin image row (y) coordinates (pixels)
            -uScale: Array of PIV vector image column (x) components (m/s)
            -vScale: Array of PIV vector image row (y) components (m/s)
            -magScale: Array of PIV vector magnitudes (m/s)
            -maskPoly: Array representing the masked out area of the outputs  
    
    Returns:
        - save_config: dictionary
            Config file keeping track of where files should be saved

    """
    # Set number of passes to 1 if not specified
    if 'passes' not in piv_params or piv_params['passes'] is None:
        piv_params['passes'] = 1

    # Set vaules for post-processing parameters
    for param in [
            'maxvel', 'minvel', 'stdThresh', 'medianFilt', 'infillFlag',
            'smoothFlag'
    ]:
        if param not in piv_params or piv_params[param] is None or piv_params[
                param] < 0:
            piv_params[param] = []

    # Check if stack has the expected field
    img_field = 'preProc'
    if img_field not in stack:
        raise ValueError(
            f"Images must be stored in a structure array field called {img_field}"
        )
    piv_params['imgField'] = img_field

    # Get the ROI polygon
    mask_poly = np.array(stack['final_roi'])

    # Calculate interrogation areas based on ideal resolution and number of passes
    piv_params['realresolution'] = np.ceil(
        piv_params['idealresolution'] /
        piv_params['pixSize']) * 2 * 2**(piv_params['passes'] - 1)
    int_areas = np.zeros(piv_params['passes'])

    for ii in range(1, piv_params['passes'] + 1):
        int_areas[ii - 1] = piv_params['realresolution'] / 2**(ii - 1)

    piv_params['intAreas'] = int_areas
    print(int_areas)
    print(
        f"Final spacing between output vectors will be: {piv_params['intAreas'][-1]*piv_params['pixSize'] /2} m"
    )
    del int_areas

    # Call ensemblePIV to perform PIV
    x_piv, y_piv, u_piv_out, v_piv_out, corr_map = ensemble_piv(
        stack, piv_params, BASE_DIR)

    # Scale vectors
    u_scale = u_piv_out * piv_params['pixSize'] / piv_params['frameInterval']
    v_scale = v_piv_out * piv_params['pixSize'] / piv_params['frameInterval']
    mag_scale = np.hypot(u_scale, v_scale)

    # SIMPLE CSV EXPORT - Save PIV results to CSV files (before masking)
    print("Saving PIV results to CSV files...")
    csv_data = []
    for i in range(x_piv.shape[0]):
        for j in range(x_piv.shape[1]):
            csv_data.append([
                x_piv[i,j], y_piv[i,j],
                u_scale[i,j], v_scale[i,j], mag_scale[i,j],
                corr_map[i,j]
            ])

    csv_df = pd.DataFrame(csv_data, columns=[
        'x_position', 'y_position',
        'u_velocity', 'v_velocity', 'velocity_magnitude',
        'correlation'
    ])
    csv_df.to_csv('piv_results.csv', index=False)
    print(f"SUCCESS: PIV results saved to piv_results.csv ({len(csv_data)} vectors)")

    # Find vectors in masked region
    x_piv_flat = x_piv.flatten()
    y_piv_flat = y_piv.flatten()
    in_poly = inpolygon(x_piv_flat, y_piv_flat, mask_poly).reshape(x_piv.shape)

    # Set indices outside of mask to nan
    x_piv[~in_poly] = np.nan
    y_piv[~in_poly] = np.nan
    u_scale[~in_poly] = np.nan
    v_scale[~in_poly] = np.nan
    mag_scale[~in_poly] = np.nan
    corr_map[~in_poly] = np.nan

    # Find current config and run directory we are working in
    directory = save_config.get('current_data_directory')
    time_stamp = os.path.basename(directory)
    output_filename = os.path.join(directory, f'{time_stamp}_PIV_output')
    os.mkdir(output_filename)

    PIVout = {
        'xPiv': x_piv,  # Velocity x position 
        'yPiv': y_piv,  # Velocity y potion 
        'uScale': u_scale,  # Velocity x magnitude
        'vScale': v_scale,  # Velocity y magnitude
        'maskPoly': mask_poly,  # Masked area
        'magScale': mag_scale,  # Velocity Magnitude
    }

    # Generate a timestamp for the filename
    for key, array in PIVout.items():
        df = pd.DataFrame(array)
        csv_file_path = f'{output_filename}/{time_stamp}_{key}.csv'
        df.to_csv(csv_file_path, index=False, header=False)

    print(f"PIVout saved as {output_filename}")

    # Move imu data to config/run directory used
    save_config_directory = save_config.get('config_folder')
    imu_data_original = os.path.join(BASE_DIR, 'save_data',
                                    save_config_directory, 'imu_data.txt')
    imu_data_new = os.path.join(directory, f'{time_stamp}_imu_data.txt')
    shutil.move(imu_data_original, imu_data_new)

    # Log to webpage PIV is finished
    with open(f'{BASE_DIR}/script.log', 'a') as log:
        log.write(f'FINISHED PIV!\n')

    return save_config


def load_config(save_config_path):
    """Load JSON configuration from a given path."""
    with open(save_config_path, 'r') as f:
        return json.load(f)


def get_filepaths(image_directory, pattern):
    """Load image paths from a directory and sort them based on a pattern."""
    image_files = sorted(os.listdir(image_directory),
                         key=lambda f: int(pattern.search(f).group(1))
                         if pattern.search(f) else float('inf'))

    return [os.path.join(image_directory, img) for img in image_files]


def main():

    # Load save configuration
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(CURRENT_DIR)
    save_config_path = os.path.join(BASE_DIR, 'save.json')

    with open(save_config_path, 'r') as f:
        save_config = json.load(f)

    # Load main configuration
    save_config_directory = save_config.get('config_folder')
    config_path = os.path.join(save_config_directory, 'config.json')
    config = load_config(config_path)

    # Compile regex pattern for sorting image files
    pattern = re.compile(r'final_frame_(\d+)\.jpg')

    # Load and sort images
    image_directory = os.path.join(BASE_DIR, 'images')
    preproc_arrays = get_filepaths(image_directory, pattern)

    # Load mask
    mask_path = os.path.join(BASE_DIR, 'app', config.get('mask_path', ''))
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    print(preproc_arrays)

    # Create stack dictionary
    stack = {
        'preProc': preproc_arrays,
        'final_roi': mask,
        'Rcrop': []  # Placeholder for geological referencing object
    }

    with open(f'{BASE_DIR}/script.log', 'a') as log:
        log.write(f'\nBegin running PIV with {len(preproc_arrays)} images.\n')

    # Load PIV parameters
    params = [
        "pixSize", "frameInterval", "minvel", "maxvel", "stdThresh",
        "medianFilt", "infillFlag", "smoothFlag", "idealresolution"
    ]
    params_dict = {key: float(config[key]) for key in params if key in config}

    # Call PIVLab and save the configuration
    save_config = call_pivlab(stack, params_dict, save_config, BASE_DIR)

    # Save updated configuration to file
    with open(save_config_path, 'w') as f:
        json.dump(save_config, f, indent=4)


if __name__ == "__main__":
    main()
