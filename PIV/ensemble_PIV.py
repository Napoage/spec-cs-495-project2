import numpy as np
import time
import cv2
from smoothn import smoothn
from inpaint_nans import inpaint_nans_spring
from post_process import post_proc
from post_process import peak_finding
import queue
import threading
import multiprocessing
"""
 Ensemble PIV Analysis using FFT-based Cross-Correlation

 Description:
 This script performs ensemble Particle Image Velocimetry (PIV) analysis using FFT-based cross-correlation
 to analyze image sequences. It processes batches of image pairs to compute velocity components and
 correlation maps, utilizing multi-threading and multiprocessing for efficiency. The final output includes 
 the PIV results (velocity components) and correlation maps, with optional post-processing such as 
 infilling and smoothing.

 Functions:
 - fft_process: Performs FFT-based cross-correlation between two images.
 - compute_subsets: Computes the subset indices for the interrogation area in the images.
 - process_image_pair: Processes an image pair by extracting subsets, performing FFT-based cross-correlation, 
   and computing the correlation map.
 - worker_thread: Worker thread function to process tasks in a task queue and store results in a result queue.
 - preload_batch: Loads images from a batch of file paths.
 - create_batches: Creates batches of image paths to optimize memory usage.
 - ensemble_piv: Core function for performing ensemble PIV analysis.

 Note:
 - The script is designed to process image pairs in batches for large image stacks.
 - Multiple processing threads are used for efficiency.
 - Optional post-processing includes infilling and smoothing of velocity fields.
 
  Credits: 
	-Funding for this project is provided through the USGS Next Generation Water Observing System (NGWOS) Research and Development program.
	-Engineered by: Deep Analytics LLC http://www.deepvt.com/
	-Ported from:
		 Legleiter, C.J., 2024, TRiVIA - Toolbox for River Velocimetry using Images from Aircraft (ver. 2.1.3, September, 2024): 
				 U.S. Geological software release, https://doi.org/10.5066/P9AD3VT3.
	-Authors: Makayla Hayes
"""

# Create a queue to hold the tasks (args) and a result queue
task_queue = queue.Queue()
result_queue = queue.Queue()


def fft_process(image1, image2):
    """
    Perform FFT-based cross-correlation between two images.

    Args:
        image1 (ndarray): The first image.
        image2 (ndarray): The second image.

    Returns:
        ndarray: The result of the cross-correlation in the frequency domain.
    """
    #split fft
    # step1 = np.fft.fft2(image1,axes = (0,1))
    # step2 = np.fft.fft2(image2,axes = (0,1))
    # step1 = np.conj(step1)
    # step3 = np.fft.ifft2(np.multiply(step1,step2),axes = (0,1))
    # step3 = np.real(step3)
    # result_conv = np.fft.fftshift(np.fft.fftshift(step3, axes=0), axes=1)

    # Combined fft - less memory used
    result_conv = np.fft.fftshift(np.fft.fftshift(np.real(
        np.fft.ifft2(np.multiply(np.conj(np.fft.fft2(image1, axes=(0, 1))),
                                 np.fft.fft2(image2, axes=(0, 1))),
                     axes=(0, 1))),
                                                  axes=0),
                                  axes=1)
    return result_conv


def compute_subsets(image1, int_area, step):
    """
    Compute the subset indices for the interrogation area in an image.

    Args:
        image1 (ndarray): The input image.
        intArea (int): The interrogation area size.
        step (float): The step size for subset extraction.

    Returns:
        tuple: A tuple containing the subset indices and other related information.
    """
    # Get indices and dimensions of the interrogation area to extract from the image
    miniy = 1 + int(np.ceil(int_area / 2))
    minix = 1 + int(np.ceil(int_area / 2))
    maxiy = int(step *
                (np.floor(image1.shape[0] / step))) - (int_area - 1) + int(
                    np.ceil(int_area / 2))
    maxix = int(step *
                (np.floor(image1.shape[1] / step))) - (int_area - 1) + int(
                    np.ceil(int_area / 2))

    numelementsy = int(np.floor((maxiy - miniy) / step + 1))
    numelementsx = int(np.floor((maxix - minix) / step + 1))

    LAy = miniy
    LAx = minix
    LUy = image1.shape[0] - maxiy
    LUx = image1.shape[1] - maxix
    shift4centery = int(round((LUy - LAy) / 2))
    shift4centerx = int(round((LUx - LAx) / 2))

    # shift4center will be negative if in the unshifted case the left border is bigger than the right border.
    if shift4centery < 0:
        shift4centery = 0

    if shift4centerx < 0:
        shift4centerx = 0

    miniy = miniy + shift4centery
    minix = minix + shift4centerx
    maxix = maxix + shift4centerx
    maxiy = maxiy + shift4centery
    pad_dim = int(np.ceil(int_area / 2))

    image1 = np.pad(image1, ((pad_dim, pad_dim), (pad_dim, pad_dim)),
                    mode='constant',
                    constant_values=np.min(image1))
    sub_pix_offset = 1 if int_area % 2 == 0 else 0.5

    # Initialize array to store output
    type_vec = np.ones((numelementsy, numelementsx), dtype=float)

    # Tile images into subsets and set up new indicies form image1 and image2
    s0 = (np.tile(
        ((np.arange(start=miniy, stop=maxiy + step, step=step) - 1).reshape(
            -1, 1)).T, (numelementsx, 1)).T + np.tile(
                ((np.arange(minix, maxix + step, step) - 1) *
                 image1.shape[0]).reshape(-1, 1), (1, numelementsy)).T).T
    s0 = s0.flatten(order='F').reshape(1, 1, -1)
    s1 = np.tile(np.arange(1, int_area + 1), (int_area, 1)).T + np.tile(
        (np.arange(1, int_area + 1) - 1) * image1.shape[0], (int_area, 1))
    ss1 = ((np.tile(s1[:, :, np.newaxis], (1, 1, s0.shape[2])) +
            np.tile(s0, (int_area, int_area, 1)))).astype(int)

    return ss1, pad_dim, type_vec, minix, maxix, miniy, maxiy, sub_pix_offset


def process_image_pair(image1, image2, ss1, ss1_flat, pad_dim, in_mask):
    """
    Process a pair of images by extracting subsets, performing FFT-based cross-correlation, 
    and computing the correlation map.

    Args:
        image1 (ndarray): The first image.
        image2 (ndarray): The second image.
        ss1 (ndarray): Subset indices for image.
        ss1_flat (ndarray): Flattened subset indices for image.
        pad_dim (int): Padding dimension for the images.
        inMask (ndarray): Mask to apply to the result.

    Returns:
        tuple: A tuple containing the cross-correlation result and the correlation map.
    """
    # Add padding
    image1 = np.pad(image1, ((pad_dim, pad_dim), (pad_dim, pad_dim)),
                    mode='constant',
                    constant_values=np.min(image1))
    image2 = np.pad(image2, ((pad_dim, pad_dim), (pad_dim, pad_dim)),
                    mode='constant',
                    constant_values=np.min(image2))

    # Flatten and cut images
    image1_flat = image1.flatten(order='F')
    image1_cut = image1_flat[ss1_flat].reshape(ss1.shape, order='F')

    image2_flat = image2.flatten(order='F')
    image2_cut = image2_flat[ss1_flat].reshape(ss1.shape, order='F')

    del image1_flat, image2_flat, image1, image2, ss1, ss1_flat

    # Run fft process
    result_conv = fft_process(image1_cut, image2_cut)
    result_conv[:, :, in_mask] = 0

    # Calculate correlation map
    corr_map = np.zeros(image1_cut.shape[2])
    for cor_i in range(image1_cut.shape[2]):
        corr = np.corrcoef(image1_cut[:, :, cor_i].flatten(order='F'),
                           image2_cut[:, :, cor_i].flatten(order='F'))[0, 1]
        corr_map[cor_i] = corr

    del corr, image1_cut, image2_cut

    return result_conv, corr_map


def worker_thread(task_queue, result_queue):
    """
    Worker thread function to process image pairs from a task queue and store the results in a result queue.

    Args:
        task_queue (queue.Queue): A queue containing tasks (image pairs and parameters) for processing.
        result_queue (queue.Queue): A queue to store the processed results (cross-correlation and correlation map).

    Returns:
        None: This function operates in place and does not return any values, but stores results in the result queue.
    """
    while not task_queue.empty():
        try:
            # Get a task (arg) from the queue
            args = task_queue.get_nowait(
            )  # Non-blocking get to avoid waiting indefinitely
        except queue.Empty:
            break

        try:
            # Process the image pair
            image1, image2, ss1, ss1_flat, pad_dim, in_mask = args
            result_conv, corr_map_part = process_image_pair(
                image1, image2, ss1, ss1_flat, pad_dim, in_mask)

            # Store the result in the result queue
            result_queue.put((result_conv, corr_map_part))
        except Exception as e:
            print(f"Error processing task: {e}")
        finally:
            task_queue.task_done()


def preload_batch(batch):
    """Load images from a batched filepaths."""
    return [
        cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) for image_path in batch
    ]


def create_batches(image_stack, batch_size):
    """Create batches of filepaths to help keep memory under control."""
    total_images = len(image_stack)
    batches = []

    # Create batches normally
    for i in range(0, total_images - 1, batch_size):
        end_index = min(i + batch_size + 1, total_images)
        batches.append(image_stack[i:end_index])

    # Handle leftover images
    if len(batches) > 1 and len(batches[-1]) == 1:
        # If the last batch has only one image, append it to the second last batch
        batches[-2].extend(batches[-1])
        batches.pop()

    return batches


def ensemble_piv(stack, piv_params, BASE_DIR):
    """
    Core function for performing ensemble PIV analysis, adapted from PIVlab. The
    ensemble correlation approach is well-suited to image sequences with low
    "seeding density" and an inhomogeneous distribution of "particles," which
    will almost always be the case in rivers. The main input is a structure
    array containing a stack of images cropped to the common area of coverage,
    masked, and pre-processed. The piv_params input specifies the pixel size and
    frame interval for the image sequence and the parameters for the PIV
    algorithm itself. The core ensemble correlation PIV functionality is adapted from PIVlab (Thielicke and Sonntag, 2021).
    
   Args:
    Stack: dictionary
            Input dictionary with variables 
                preProc - dictionary of image paths
                finalROi - a mask delineating the ROI for the PIV analysis
                Rcrop - []
    piv_params: dictionary 
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
        x_piv: Array of PIV vector origin image column (x) coordinates (pixels)
        y_piv: Array of PIV vector origin image row (y) coordinates (pixels)
        u_piv_out: Array of PIV vector column (x) velocity components (pix/frames)
        v_piv_out: Array of PIV vector image row (y) velocity components (pix/frames)
        corr_map: Array with peak correlation values for each interrogation areas
        
    Notes:

    """

    passes = piv_params['passes']
    int_areas = piv_params['intAreas']

    # Check int_area is less than passes
    if len(piv_params['intAreas']) < piv_params['passes']:
        raise ValueError(
            'Number of elements in the intAreas input must be >= specified number of passes'
        )

    # Set intarea to int calculate step
    int_area = int(int_areas[0])  #probably can move
    step = int_area / 2

    if passes == 1:
        int2, int3 = [], []

    # Start an overall timer
    tStart = time.time()

    img_field = piv_params['imgField']

    ss1, pad_dim, type_vec, minix, maxix, miniy, maxiy, sub_pix_offset = compute_subsets(
        cv2.imread(stack['preProc'][0], cv2.IMREAD_GRAYSCALE), int_area, step)
    ss1_flat = ss1.flatten(order='F') - 1

    mask = np.zeros_like(cv2.imread(stack['preProc'][0], cv2.IMREAD_GRAYSCALE))
    mask = np.pad(mask, ((pad_dim, pad_dim), (pad_dim, pad_dim)),
                  mode='constant',
                  constant_values=0)

    # Initialize result containers
    result_conv_ensemble = None
    corr_map = None
    corr_map_cnt = 0

    # Prepare arguments for multiprocessing
    fftst = time.time()
    ss1int = ss1[round(int_area // 2),
                 round(int_area // 2), :].reshape(1, 1, -1)

    # Apply
    in_mask = mask.flatten(order='F')[ss1int.flatten(order='F') - 1].reshape(
        ss1int.shape, order='F')

    if np.all(in_mask == 0):
        in_mask = np.ones_like(in_mask)

    # Batch images
    batch_number = 0
    batches = create_batches(stack['preProc'], 50)

    # Log to webpage
    with open(f'{BASE_DIR}/script.log', 'a') as log:
        log.write(
            f'{len(batches)} batches of 50 images will be run through FFT process\n'
        )

    # Loop through batches
    for batch in batches:

        threads = []
        results = []
        batch_start = time.time()
        batch_number += 1

        # Log to spec_app.log
        print("on batch number ", batch_number)

        # Log batch number to webpage
        with open(f'{BASE_DIR}/script.log', 'a') as log:
            log.write(f'Running batch {batch_number} of {len(batches)}\n')
        prealoaded_time = time.time()
        preloaded_images = preload_batch(batch)

        # Log to spec_app.log
        print('preload time ', time.time() - prealoaded_time)

        # Directly populate the task queue with the arguments
        for ii in range(len(preloaded_images) - 1):
            image1 = preloaded_images[ii]
            image2 = preloaded_images[ii + 1]
            task_queue.put((image1, image2, ss1, ss1_flat, pad_dim, in_mask))

        # Determine the number of processing threads
        processing_threads = min(multiprocessing.cpu_count() - 1,
                                 task_queue.qsize())

        # Create and start worker threads
        for _ in range(processing_threads):
            t = threading.Thread(target=worker_thread,
                                 args=(task_queue, result_queue))
            t.start()
            threads.append(t)

        # Wait for all tasks in the queue to be processed
        task_queue.join()

        # Wait for all threads to finish
        for t in threads:
            t.join()

        # Log to webpage
        with open(f'{BASE_DIR}/script.log', 'a') as log:
            log.write(
                f'Adding results from batch {batch_number} to total results\n')

        # Aggregate results from the result queue
        while not result_queue.empty():
            results.append(result_queue.get())

        # Log to spec_app.log)
        results_start = time.time()
        if corr_map is None and result_conv_ensemble is None:
            result_conv_ensemble = np.zeros(
                results[0][0].shape).flatten(order='F')
            corr_map = np.zeros(results[0][1].shape).flatten(order='F')

        # Sum up the results in batch
        for result_conv, corr_map_part in results:
            result_conv_ensemble += result_conv.flatten(order='F')
            corr_map += corr_map_part.flatten(order='F')
            corr_map_cnt += 1

        # Log to spec_app.log
        print('results process time ', time.time() - results_start)

    # Reshape results_conv_ensember and corr_map
    result_conv_ensemble = result_conv_ensemble.reshape(results[0][0].shape,
                                                        order='F')
    corr_map = corr_map.reshape(results[0][1].shape, order='F')

    # Log to spec_app.log
    print(f'FFT time {time.time() - fftst}')

    # Log to webpage
    with open(f'{BASE_DIR}/script.log', 'a') as log:
        log.write(f'Finished FFT process for all image batches\n')

    # Find peaks
    pst = time.time()
    x_piv, y_piv, u_piv, v_piv = peak_finding(result_conv_ensemble,
                                              mask,
                                              int_area,
                                              minix,
                                              step,
                                              maxix,
                                              miniy,
                                              maxiy,
                                              sub_pix_offset,
                                              ss1,
                                              sub_pix_finder=1)

    # Log to spec_app.log
    print(f'peak finding time {time.time()- pst}')

    # Log to webpage
    with open(f'{BASE_DIR}/script.log', 'a') as log:
        log.write(f'Post Processing!\n')

    post = time.time()
    u_piv, v_piv = post_proc(u_piv, v_piv, piv_params, BASE_DIR)

    # Log to spec_app.log
    print(f'post poc time= {time.time()-post}')

    # Replace NaNs using PIVlab function inpaint_nans.m
    if piv_params['infillFlag'] and piv_params['infillFlag'] > 0:
        # Log to spec_app.log
        print(
            "**** Infilling gaps in initial output from ensemble PIV algorithm ****"
        )
        inp = time.time()
        uPivInfill = inpaint_nans_spring(u_piv)
        vPivInfill = inpaint_nans_spring(v_piv)
        # Log to spec_app.log
        print(f'inpaint_nans = {time.time()-inp}')

    # Smooth final PIV output using PIVlab function smoothn.m
    if piv_params['smoothFlag'] and piv_params['smoothFlag'] > 0:
        # Log to spec_app.log
        print("**** Smoothing initial output from ensemble PIV algorithm ****")
        if 'uPivInfill' in locals() and 'vPivInfill' in locals():
            smst = time.time()
            uPivSmooth = smoothn(uPivInfill, s=0.05)
            vPivSmooth = smoothn(vPivInfill, s=0.05)
            # Log to spec_app.log
            print(f'smooth time = {time.time()-smst}')
        else:
            uPivSmooth = smoothn(u_piv, s=0.05)
            vPivSmooth = smoothn(v_piv, s=0.05)

    # Update output variables depending on what level of post-processing we've done
    if 'uPivInfill' in locals() and 'uPivSmooth' in locals():
        # Log to spec_app.log
        print("Exporting infilled and smoothed output")
        u_piv_out = uPivSmooth
        v_piv_out = vPivSmooth
    elif 'uPivInfill' in locals() and 'uPivSmooth' not in locals():
        # Log to spec_app.log
        print("Exporting infilled but not smoothed output")
        u_piv_out = uPivInfill
        v_piv_out = vPivInfill
    else:
        # Log to spec_app.log
        print(
            "No infilling or smoothing applied, so exporting output without these procedures applied"
        )
        u_piv_out = u_piv
        v_piv_out = v_piv

    # Update coordinate grids
    x_piv = x_piv - int(np.ceil(int_area / 2))
    y_piv = y_piv - int(np.ceil(int_area / 2))

    # Obtain final overall correlation map, averaged over the frame pairs
    corr_map = np.transpose(np.reshape(corr_map, (np.flip(x_piv.shape))),
                            axes=[1, 0]) / corr_map_cnt

    # Clear correlation map in masked area
    corr_map[type_vec == 0] = 0
    corr_map = np.nan_to_num(corr_map)

    # Summarize run
    runTime = time.time() - tStart

    # Log to webpage
    with open(f'{BASE_DIR}/script.log', 'a') as log:
        log.write(f'Ensemble PIV completed in {runTime:.2f} s\n')

    # Log to spec_app.log
    print(f"Ensemble PIV completed in {runTime:.2f} s")

    return x_piv, y_piv, u_piv_out, v_piv_out, corr_map
