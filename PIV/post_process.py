import numpy as np
from scipy.ndimage import median_filter
import cv2
# from inpaint_nans import inpaint_simple
from scipy.signal import medfilt2d
from medfilt2 import medfilt2
from scipy.ndimage import median_filter
from scipy.ndimage import label
from scipy.ndimage import distance_transform_edt
import time
"""
 Peak Finding and Sub-Pixel Interpolation for PIV (Particle Image Velocimetry)

 This script contains functions to perform peak finding on correlation surfaces generated
 from an ensemble of images and calculate sub-pixel velocity components using a Gaussian peak 
 fitting method. It includes post-processing functions for filtering and refining PIV vectors.

 Functions:
 - peak_finding: Identifies peaks from a correlation surface and extracts PIV vectors
 - SUBPIXGAUSS: Performs sub-pixel interpolation to refine the PIV velocity components
 - post_proc: Post-processes the PIV results by applying velocity magnitude limits, local 
   median checks, and standard deviation filtering.
   
 Credits: 
	-Funding for this project is provided through the USGS Next Generation Water Observing System (NGWOS) Research and Development program.
	-Engineered by: Deep Analytics LLC http://www.deepvt.com/
	-Ported from:
		 Legleiter, C.J., 2024, TRiVIA - Toolbox for River Velocimetry using Images from Aircraft (ver. 2.1.3, September, 2024): 
				 U.S. Geological software release, https://doi.org/10.5066/P9AD3VT3.
	-Authors: Makayla Hayes

"""


def peak_finding(result_conv_ensemble, mask, int_area, minix, step, maxix,
                 miniy, maxiy, sub_pix_offset, ss1, sub_pix_finder):
    """
    Helper function to find the peaks of a correlation surface. By normalizing the results conv ensemble and finding the 
    points where 255 exits and extracting one peak from each couple peaks. 
    
    Args:
    
        -results_conv_ensemble: array 
            Arry of correlation areas stacked on eachother
        -mask: array 
            Arry of zeros
        -intArea: int 
            Size of interrogation array
        -minx: int 
            Minimum x value of interrogation area
        -step: int 
            Step between pixels of interrogation area
        -miniy: int 
            Minimum y value of interrogation area
        -maxix: int 
            Maximum x value of interrogation area
        -maxiy: int 
            Maximum y value of interrogation area
        -SubPixoffeset float 
            Offset between pixels either 1 or 0.5
        -ss1: array 
            Of indicies extracted from images for interrogation area
        -subpixfinder: int 
            Flag to specify three point gausian peak finding
    
    Returns:
        -x_piv: array
            Array of PIV vector origin image column (x) coordinates (pixels)
        -y: array
            Array of PIV vector (y) coordinates (pixels)
        u: array
            Array of PIV vector column (x) velocity components
        v: array
            Array of PIV vector row (y) velocity components
    
    """
    minres = np.tile(
        np.min(result_conv_ensemble, axis=(0, 1), keepdims=True),
        (result_conv_ensemble.shape[0], result_conv_ensemble.shape[1], 1))
    deltares = np.tile(
        np.max(result_conv_ensemble, axis=(0, 1), keepdims=True) -
        np.min(result_conv_ensemble, axis=(0, 1), keepdims=True),
        (result_conv_ensemble.shape[0], result_conv_ensemble.shape[1], 1))

    # Prevent division by zero if deltares is zero
    result_conv_ensemble = ((result_conv_ensemble - minres) / deltares) * 255

    # Apply mask
    if mask.size > 0:
        if np.all(mask == 0):
            mask = np.ones_like(mask)

        iiint = ss1[round(int_area // 2),
                    round(int_area // 2), :].reshape(1, 1, -1)
        ii = mask.flatten(order='F')[iiint.flatten(order='F') - 1].reshape(
            iiint.shape, order='F')

        # Change ii values to 0 and nans to 0 to follow matlab layout
        result_conv_ensemble[:, :, ii] = 0
        result_conv_ensemble = np.nan_to_num(result_conv_ensemble)

    indices = np.where(result_conv_ensemble == 255)
    y, x_piv, z = indices

    # Extract only one peak from each couple of pictures
    zi = np.argsort(z)
    z1 = z[zi]
    dz1 = np.concatenate([[z1[0]], np.diff(z1)])
    i0 = np.where(dz1 != 0)[0]

    # Final results
    x1 = x_piv[zi[i0]]
    y1 = y[zi[i0]]
    z1 = z[zi[i0]]

    # Array Construction
    x_range = np.arange(minix, maxix + step, step)
    y_range = np.arange(miniy, maxiy + step, step)
    x_piv = np.tile(x_range.T + int_area / 2, (len(y_range), 1))
    y = np.tile((y_range + int_area / 2).reshape(-1, 1), (1, len(x_range)))

    if sub_pix_finder == 1:
        st = time.time()
        vector = SUBPIXGAUSS(result_conv_ensemble, int_area, x1, y1, z1,
                             sub_pix_offset)
        print(f'subpix = {time.time()-st}')

    vector = np.transpose(
        vector.reshape((x_piv.shape[1], x_piv.shape[0], 2), order='F'),
        (1, 0, 2))
    u, v = vector[:, :, 0], vector[:, :, 1]

    return x_piv, y, u, v


def SUBPIXGAUSS(result_conv, int_area, x_piv, y, z, sub_pix_offset):
    """
    Helper function for sub pixel peak finding
    Args:
        -results_conv: array
            array of correlation areas of images normalize?
        -intArea int
            size of integration area
        -x_Piv: array
            array of PIV vector (x) coordinates
        -y: array
            array of PIV vector (y) coordinates
        -z array
            array of PIV vector (z) coordinates
        -sub_pix_offset int
            offset between pixels either 1 or 0.5
    Returns:
        -Vector: array
            x,y velocity components 
    """
    height, width, depth = result_conv.shape

    # Find indices where conditions are not satisfied and remove them
    xi = np.where(~((x_piv <= (result_conv.shape[1] - 2)) &
                    (y <= (result_conv.shape[0] - 2)) & (x_piv >= 1) &
                    (y >= 1)))
    x_piv = np.delete(x_piv, xi)
    y = np.delete(y, xi)
    z = np.delete(z, xi)

    vector = np.full((depth, 2), np.nan)

    # Filter out invalid indices
    # valid_indices = (xPiv > 0) & (xPiv < width - 1) & (y > 0) & (y < height - 1)
    # xPiv = xPiv[valid_indices]
    # y = y[valid_indices]
    # z = z[valid_indices]

    if len(x_piv) != 0:
        ip = (np.ravel_multi_index((y, x_piv, z),
                                   dims=result_conv.shape,
                                   order='F'))
        flatt = result_conv.flatten(order='F')

        f0 = np.log(flatt[ip])
        f1 = np.log(flatt[ip - 1])
        f2 = np.log(flatt[ip + 1])

        peaky = y + (f1 - f2) / (2 * f1 - 4 * f0 + 2 * f2)

        f0 = np.log(flatt[ip])
        f1 = np.log(flatt[ip - width])
        f2 = np.log(flatt[ip + width])

        peakx = x_piv + (f1 - f2) / (2 * f1 - 4 * f0 + 2 * f2)

        sub_pixelX = (peakx - (int_area / 2) - sub_pix_offset) + 1
        sub_pixelY = (peaky - (int_area / 2) - sub_pix_offset) + 1
        vector[z, 0] = sub_pixelX
        vector[z, 1] = sub_pixelY

    return vector


def post_proc(u, v, piv_params, BASE_DIR):
    """
    Helper function for vector post-processing (filtering)
    Args:
        -u: array
            array of vector column (x) velocity components
        -v: array of vector row (y) velocity components
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
    
    Returns:
        -u_filt: array
            array of filtered vector column (x) velocity components
        -v_filt: array
            array of filtered vector row (y) velocity components
    """
    print("**** Filtering initial output from ensemble PIV algorithm ****")

    # Calculate conversion factor from pixels/frame to meters/second and scale vectors
    scale = piv_params['pixSize'] / piv_params['frameInterval']
    u_scale = u * scale
    v_scale = v * scale
    mag_scale = np.hypot(u_scale, v_scale)

    # Velocity magnitude limits
    if piv_params['minvel'] is not None:
        below_threshold_indices = mag_scale < piv_params['minvel']
        u_scale[mag_scale < piv_params['minvel']] = np.nan
        v_scale[mag_scale < piv_params['minvel']] = np.nan
        num_below_threshold = np.sum(below_threshold_indices)
        # output = f"Reset {num_below_threshold} vectors below minimum velocity threshold to NaN\n"
        print(
            f"Reset {num_below_threshold} vectors below minimum velocity threshold to NaN"
        )

    if piv_params['maxvel'] is not None:
        above_threshold_indices = mag_scale > piv_params['maxvel']
        u_scale[mag_scale > piv_params['maxvel']] = np.nan
        v_scale[mag_scale > piv_params['maxvel']] = np.nan
        num_above_threshold = np.sum(above_threshold_indices)
        print(
            f"Reset {num_above_threshold} vectors above maximum velocity threshold to NaN"
        )

    # Local median check
    if piv_params['medianFilt'] is not None:
        # First for u component
        neigh_filt = medfilt2(u_scale)

        # Create a mask where NaN values are marked as True
        mask = np.isnan(neigh_filt)

        try:
            # Inpaint NaNs
            inpaint_region = cv2.inpaint(neigh_filt.astype(np.uint8),
                                         (mask * 255).astype(np.uint8), 3,
                                         cv2.INPAINT_TELEA)
            neigh_filt[mask] = inpaint_region[mask]
        except:
            neigh_filt = np.full_like(neigh_filt, np.nan)

        neigh_diff = np.abs(neigh_filt - u_scale)
        u_scale[neigh_diff > piv_params['medianFilt']] = np.nan
        num_failed = np.sum(neigh_diff > piv_params['medianFilt'])
        print(
            f"Reset {num_failed} u components that failed local median check to NaN"
        )

        # Same kind of thing for v component
        neigh_filt = medfilt2(v_scale)

        # Create a mask where NaN values are marked as True
        mask = np.isnan(neigh_filt)

        try:
            # Inpaint NaNs
            inpaint_region = cv2.inpaint(neigh_filt.astype(np.uint8),
                                         (mask * 255).astype(np.uint8), 3,
                                         cv2.INPAINT_TELEA)
            neigh_filt[mask] = inpaint_region[mask]
        except:
            neigh_filt = np.full_like(neigh_filt, np.nan)

        neigh_diff = np.abs(neigh_filt - v_scale)
        v_scale[neigh_diff > piv_params['medianFilt']] = np.nan
        num_failed = np.sum(neigh_diff > piv_params['medianFilt'])
        print(
            f"Reset {num_failed} v components that failed local median check to NaN"
        )
        #u_scale, v_scale = reset_components(u_scale, v_scale, piv_params['medianFilt'])

    # Standard deviation check
    if piv_params['stdThresh'] is not None:
        meanu = np.nanmean(u_scale)
        meanv = np.nanmean(v_scale)
        stdu = np.nanstd(u_scale)
        stdv = np.nanstd(v_scale)

        std_threshold = piv_params['stdThresh'] * stdu
        minvalu = meanu - std_threshold
        maxvalu = meanu + std_threshold

        std_threshold = piv_params['stdThresh'] * stdv
        minvalv = (meanv - std_threshold)
        maxvalv = (meanv + std_threshold)

        # Set values outside the threshold to NaN
        u_scale_low_mask = u_scale < minvalu
        u_scale_high_mask = u_scale > maxvalu
        v_scale_low_mask = v_scale < minvalv
        v_scale_high_mask = v_scale > maxvalv

        u_scale[u_scale_low_mask | u_scale_high_mask] = np.nan
        v_scale[v_scale_low_mask | v_scale_high_mask] = np.nan

        # Print messages
        num_u_low = np.count_nonzero(u_scale_low_mask)
        num_u_high = np.count_nonzero(u_scale_high_mask)
        num_v_low = np.count_nonzero(v_scale_low_mask)
        num_v_high = np.count_nonzero(v_scale_high_mask)

        # Log changed values
        print(
            f"Reset {num_u_low} u components that failed standard deviation check (too low) to NaN"
        )
        print(
            f"Reset {num_u_high} u components that failed standard deviation check (too high) to NaN"
        )
        print(
            f"Reset {num_v_low} v components that failed standard deviation check (too low) to NaN"
        )
        print(
            f"Reset {num_v_high} v components that failed standard deviation check (too high) to NaN"
        )

    # Force any vector that has a nan for one component to have a nan for the other component as well
    u_scale[np.isnan(v_scale)] = np.nan
    v_scale[np.isnan(u_scale)] = np.nan

    # Output resulting filtered vectors after unapplying the scale
    u_filt = u_scale / scale
    v_filt = v_scale / scale

    return u_filt, v_filt
