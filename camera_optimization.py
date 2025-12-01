import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

video_path = r"VideoAugmentation/Water Moving_lighten_full_lighten_full.mp4"

def brighten_video(video_path, intensity):
    """
    Brighten a video by increasing pixel intensity values across all frames.

    Processes each frame of the video by adding a constant intensity value to all
    pixels, then saves the result back to the original file path.

    Args:
        video_path (str): Path to the video file to be brightened. The original
            file will be replaced with the brightened version.
        intensity (int): Amount to increase pixel intensity values. Positive values
            brighten the video. Typical range is 0-100.

    Notes:
        - Processes all frames in the video sequentially
        - Uses cv2.convertScaleAbs with beta parameter for brightness adjustment
        - Modifies the original file in place (creates temporary file during processing)
        - Preserves original video dimensions, frame rate, and codec (mp4v)
        - Prints progress messages during processing
    """
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print("Error: Could not open video.")
        exit()

    fps = int(video.get(cv2.CAP_PROP_FPS))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Brightening video...")

    temp_path = video_path.replace('.mp4', '_temp.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))

    while True:
        ret, frame = video.read()
        if not ret: 
            print("Brightening complete.")
            break
        
        processed_frame = frame.copy()
        h = processed_frame.shape[0]

        processed_frame = cv2.convertScaleAbs(processed_frame, alpha=1.0, beta=intensity)
        out.write(processed_frame)

    video.release()
    out.release()
    cv2.destroyAllWindows()

    os.remove(video_path)  # Delete original
    os.rename(temp_path, video_path)  # Rename temp to original name

    print(f"Video brightened.")

def darken_video(video_path, intensity):
    """
    Darken a video by decreasing pixel intensity values across all frames.

    Processes each frame of the video by adding a constant intensity value to all
    pixels, then saves the result back to the original file path.

    Args:
        video_path (str): Path to the video file to be brightened. The original
            file will be replaced with the brightened version.
        intensity (int): Amount to increase pixel intensity values. Positive values
            brighten the video. Typical range is 0-100.

    Notes:
        - Processes all frames in the video sequentially
        - Uses cv2.convertScaleAbs with beta parameter for brightness adjustment
        - Modifies the original file in place (creates temporary file during processing)
        - Preserves original video dimensions, frame rate, and codec (mp4v)
        - Prints progress messages during processing
    """
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print("Error: Could not open video.")
        exit()

    print(f"Darkening video...")

    fps = int(video.get(cv2.CAP_PROP_FPS))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    temp_path = video_path.replace('.mp4', '_temp.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))

    while True:
        ret, frame = video.read()
        if not ret: 
            print("Darkening complete.")
            break
        
        processed_frame = frame.copy()
        h = processed_frame.shape[0]

        processed_frame = cv2.convertScaleAbs(processed_frame, alpha=0.7, beta=intensity)
        out.write(processed_frame)

    video.release()
    out.release()
    cv2.destroyAllWindows()

    os.remove(video_path)  # Delete original
    os.rename(temp_path, video_path)  # Rename temp to original name

    print(f"Video darkened.")

def check_exposure(image):
    """
    Check an image for overexposed and underexposed pixels.
    
    Analyzes the grayscale intensity distribution to identify the percentage
    of pixels that are too dark or too bright, which may indicate exposure
    problems.
    
    Args:
        image (numpy.ndarray): The input image.
    
    Returns:
        tuple[float, float, float, float]: A tuple containing:
            - dark_pixels (float): Percentage of pixels with intensity 0-50.
            - bright_pixels (float): Percentage of pixels with intensity 200-255.
            - clipped_black (float): Percentage of pixels with intensity ≤ 5 (near pure black).
            - clipped_white (float): Percentage of pixels with intensity ≥ 250 (near pure white).
    """
    #convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #create histogram for grayscale image. hist[i] = # of pixels with intensity i
    hist = cv2.calcHist([gray], [0], None, [256], [0,256])

    #get total # of pixels by multiplying width and height
    total_pixels = gray.shape[0] * gray.shape[1]
    #get percentage of dark pixels, defined as intensity 0 to 50
    dark_pixels = sum(hist[:50]) / total_pixels
    #get percentage of bright pixels, defined as intensity 200 to 256
    bright_pixels = sum(hist[200:256]) / total_pixels

    #get percentage of pixels that are close to pure black
    clipped_black = np.sum((gray <= 5).astype(np.uint64)) / total_pixels
    #get percentage of pixels that are close to pure white
    clipped_white = np.sum((gray >= 250).astype(np.uint64)) / total_pixels

    return dark_pixels, bright_pixels, clipped_black, clipped_white

def process_video(video_path):
    """
    Iteratively adjust a video's exposure until it meets acceptable quality thresholds.

    Analyzes video frames for exposure issues (too dark, too bright, clipped pixels)
    and automatically corrects them by brightening or darkening the video. Repeats
    the process up to 10 times or until the video meets quality standards.

    Args:
        video_path (str): Path to the video file to be processed. The file will be
            modified in place if adjustments are needed.

    Notes:
        - Samples every 10th frame (FRAME_SKIP = 10) for analysis
        - Maximum of 10 correction iterations to prevent infinite loops
        - Adjusts brightness by ±50 units per iteration
        - Video is considered acceptable when:
            * ≤50% of frames are too dark (>60% dark pixels)
            * ≤50% of frames are too bright (>40% bright pixels)
            * ≤30% of frames have clipped blacks (>5% clipped black pixels)
            * ≤30% of frames have clipped whites (>10% clipped white pixels)
        - Prints a summary after each iteration showing frame counts and percentages
        - Modifies the original video file with each adjustment
    """
    iteration = 0
    max_iterations = 10

    while iteration < max_iterations:
        print(f"\n=== Iteration {iteration + 1} ===")

        # Open the video file
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("Error: Could not open video.")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_count = 0

        # number of frames that are too bright or too dark overall
        overall_dark_count = 0
        overall_bright_count = 0
        # number of frames that have too much clipped black or white
        clipped_black_count = 0
        clipped_white_count = 0

        FRAME_SKIP = 10

        last_full_frame = (total_frames // FRAME_SKIP) * FRAME_SKIP

        while True:
            ret, frame = cap.read()
            if not ret or frame_count == last_full_frame:
                break  # Break the loop if there are no more frames

            frame_count += 1  # counts *all* frames in the video

            # only process every 10th frame
            if frame_count % FRAME_SKIP != 0:
                continue  

            # Resize the frame for faster processing (optional)
            #frame = cv2.resize(frame, (640, 480))

            # check exposure levels
            dark_pixels, bright_pixels, clipped_black, clipped_white = check_exposure(frame)

            if dark_pixels > .6:
                overall_dark_count += 1

            if bright_pixels > .4:
                overall_bright_count += 1

            if clipped_black > 0.05:
                clipped_black_count += 1

            if clipped_white > .1:
                clipped_white_count += 1    

            # Optional: Display the frame (press 'q' to quit)
            # cv2.imshow("Frame", frame)
            # if cv2.waitKey(1) & 0xFF == ord("q"):
            #     break

        # Release the video capture object and close all OpenCV windows
        cap.release()
        #cv2.destroyAllWindows()
        print(f"\n--- Summary ---")
        print(f"Total frames: {frame_count}")

        print(f"Overall dark frames: {overall_dark_count * FRAME_SKIP} ({overall_dark_count * FRAME_SKIP/frame_count*100:.1f}%)")
        print(f"Overall bright frames: {overall_bright_count * FRAME_SKIP} ({overall_bright_count * FRAME_SKIP/frame_count*100:.1f}%)")
        print(f"Clipped black frames: {clipped_black_count * FRAME_SKIP} ({clipped_black_count * FRAME_SKIP/frame_count*100:.1f}%)") 
        print(f"Clipped white frames: {clipped_white_count * FRAME_SKIP} ({clipped_white_count * FRAME_SKIP/frame_count*100:.1f}%)")
        
        overall_dark_percentage = overall_dark_count * FRAME_SKIP / frame_count
        overall_bright_percentage = overall_bright_count * FRAME_SKIP / frame_count
        clipped_black_percentage = clipped_black_count * FRAME_SKIP / frame_count
        clipped_white_percentage = clipped_white_count * FRAME_SKIP / frame_count

        if (overall_dark_percentage > .5) or (clipped_black_percentage > .3):
            brighten_video(video_path, 50)
            iteration += 1

        elif(overall_bright_percentage > .5) or (clipped_white_percentage > .3):
            darken_video(video_path, 50)
            iteration += 1
        else:
            print(f"Video exposure is now acceptable")
            break
        
        if iteration >= max_iterations:
            print(f"Reached maximum iterations.")
            break

if __name__ == "__main__":
    flag = process_video(video_path)



