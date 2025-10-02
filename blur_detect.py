import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def is_blurry(image, threshold=1000.0):
    """
    Detect if an image is blurry using the Laplacian variance method.

    Args:
        image (numpy.ndarray): The input image.
        threshold (float): Variance threshold below which the image is considered blurry.

    Returns:
        bool: True if the image is blurry, False otherwise.
        float: The variance of the Laplacian.
    """
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute the Laplacian of the image
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)

    # Compute the variance of the Laplacian
    variance = laplacian.var()

    # Determine if the image is blurry
    return laplacian, variance < threshold, variance

def check_exposure(image):
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
    clipped_black = np.sum(gray <= 5) / total_pixels
    #get percentage of pixels that are close to pure white
    clipped_white = np.sum(gray >= 250) / total_pixels

    return dark_pixels, bright_pixels, clipped_black, clipped_white

def generate_score(frame_count,  sharp_frames, avg_variance, overall_dark_count, overall_bright_count, clipped_black_count, clipped_white_count):

    sharp_frames_percentage = sharp_frames / frame_count
    overall_dark_percentage = overall_dark_count / frame_count
    overall_bright_percentage = overall_bright_count / frame_count
    clipped_black_percentage = clipped_black_count / frame_count
    clipped_white_percentage = clipped_white_count / frame_count

    sharp_frames_score = sharp_frames_percentage * 16.666666666

    if avg_variance >= 2500:
        variance_score = 16.666666666
    else:
        variance_score = (avg_variance / 2500) * 16.666666666
    
    bright_frames_score = (1 - overall_bright_percentage) * 16.666666666
    dark_frames_score = (1 - overall_dark_percentage) * 16.666666666
    clipped_black_score = (1 - clipped_black_percentage) * 16.666666666
    clipped_white_score = (1 - clipped_white_percentage) * 16.666666666

    if sharp_frames_percentage < 0.3:
        total_score = 0
    elif variance_score < 5:
        total_score = 0
    elif overall_dark_percentage > .6:
        total_score = 0
    elif overall_bright_percentage > .6: 
        total_score = 0
    elif clipped_black_percentage > .5:  
        total_score = 0
    elif clipped_white_percentage > .5:
        total_score = 0
    else:
        total_score = sharp_frames_score + variance_score + bright_frames_score + dark_frames_score + clipped_black_score + clipped_white_score
    
    return total_score

def process_video(video_path, threshold=1000.0):
    """
    Process each frame of a video to check if it's blurry.

    Args:
        video_path (str): Path to the video file.
        threshold (float): Variance threshold below which a frame is considered blurry.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0
    blurry_frames = 0
    sharp_frames = 0 
    variance_list = []

    # number of frames that are too bright or too dark overall
    overall_dark_count = 0
    overall_bright_count = 0
    # number of frames that have too much clipped black or white
    clipped_black_count = 0
    clipped_white_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Break the loop if there are no more frames

        frame_count += 1

        # Resize the frame for faster processing (optional)
        #frame = cv2.resize(frame, (640, 480))

        # Check if the frame is blurry
        _, blurry, variance = is_blurry(frame, threshold=threshold)

        if blurry:
            blurry_frames += 1
        else:
            sharp_frames += 1

        variance_list.append(variance)

        # check exposure levels
        dark_pixels, bright_pixels, clipped_black, clipped_white = check_exposure(frame)

        if dark_pixels > .6:
            overall_dark_count += 1

        if bright_pixels > .4:
            overall_bright_count += 1

        if clipped_black > .1:
            clipped_black_count += 1

        if clipped_white > .1:
            clipped_white_count += 1    
        


    
        # print(
        #     f"Frame {frame_count} is {'blurry' if blurry else 'sharp'} (Variance: {variance:.2f})"
        # )

        # Optional: Display the frame (press 'q' to quit)
        # cv2.imshow("Frame", frame)
        # if cv2.waitKey(1) & 0xFF == ord("q"):
        #     break

    # Release the video capture object and close all OpenCV windows
    cap.release()
    #cv2.destroyAllWindows()
    print(f"\n--- Summary ---")
    print(f"Total frames: {frame_count}")
    print(f"Blurry frames: {blurry_frames} ({blurry_frames/frame_count*100:.1f}%)")
    print(f"Sharp frames: {sharp_frames} ({sharp_frames/frame_count*100:.1f}%)")

    avg_variance = sum(variance_list) / len(variance_list)
    print(f"Average Variance: {avg_variance:.2f}")

    print(f"Overall dark frames: {overall_dark_count} ({overall_dark_count/frame_count*100:.1f}%)")
    print(f"Overall bright frames: {overall_bright_count} ({overall_bright_count/frame_count*100:.1f}%)")
    print(f"Clipped black frames: {clipped_black_count} ({clipped_black_count/frame_count*100:.1f}%)") 
    print(f"Clipped white frames: {clipped_white_count} ({clipped_white_count/frame_count*100:.1f}%)")
    total_score = generate_score(frame_count, sharp_frames, avg_variance, overall_dark_count, overall_bright_count, clipped_black_count, clipped_white_count)
    print(f"Overall video quality score (out of 100): {total_score:.2f}")


video_path = (
    r"VideoAugmentation/ACS_blur_half.mp4"  # Replace with your video path
)
process_video(video_path, threshold=1250.0)