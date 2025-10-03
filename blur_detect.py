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
    clipped_black = np.sum((gray <= 5).astype(np.uint64)) / total_pixels
    #get percentage of pixels that are close to pure white
    clipped_white = np.sum((gray >= 250).astype(np.uint64)) / total_pixels

    return dark_pixels, bright_pixels, clipped_black, clipped_white

def generate_score(frame_count,  sharp_frames, avg_variance, overall_dark_count, overall_bright_count, clipped_black_count, clipped_white_count, FRAME_SKIP):

    sharp_frames_percentage = sharp_frames * FRAME_SKIP / frame_count
    overall_dark_percentage = overall_dark_count * FRAME_SKIP / frame_count
    overall_bright_percentage = overall_bright_count * FRAME_SKIP / frame_count
    clipped_black_percentage = clipped_black_count * FRAME_SKIP / frame_count
    clipped_white_percentage = clipped_white_count * FRAME_SKIP / frame_count

    WEIGHT = 100 / 6

    sharp_frames_score = sharp_frames_percentage * WEIGHT
    variance_score = min(avg_variance / 2500, 1.0) * WEIGHT
    bright_frames_score = (1 - overall_bright_percentage) * WEIGHT
    dark_frames_score = (1 - overall_dark_percentage) * WEIGHT
    clipped_black_score = (1 - clipped_black_percentage) * WEIGHT
    clipped_white_score = (1 - clipped_white_percentage) * WEIGHT

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

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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

        if clipped_black > 0.05:
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
    print(f"Blurry frames: {blurry_frames * FRAME_SKIP} ({blurry_frames * FRAME_SKIP/frame_count*100:.1f}%)")
    print(f"Sharp frames: {sharp_frames * FRAME_SKIP} ({sharp_frames * FRAME_SKIP/frame_count*100:.1f}%)")

    avg_variance = sum(variance_list) / len(variance_list)
    print(f"Average Variance: {avg_variance:.2f}")

    print(f"Overall dark frames: {overall_dark_count * FRAME_SKIP} ({overall_dark_count * FRAME_SKIP/frame_count*100:.1f}%)")
    print(f"Overall bright frames: {overall_bright_count * FRAME_SKIP} ({overall_bright_count * FRAME_SKIP/frame_count*100:.1f}%)")
    print(f"Clipped black frames: {clipped_black_count * FRAME_SKIP} ({clipped_black_count * FRAME_SKIP/frame_count*100:.1f}%)") 
    print(f"Clipped white frames: {clipped_white_count * FRAME_SKIP} ({clipped_white_count * FRAME_SKIP/frame_count*100:.1f}%)")
    total_score = generate_score(frame_count, sharp_frames, avg_variance, overall_dark_count, overall_bright_count, clipped_black_count, clipped_white_count, FRAME_SKIP)
    print(f"Overall video quality score (out of 100): {total_score:.2f}")

    return total_score

if __name__ == "__main__":
    video_path = r"VideoAugmentation/ACS_lighten_full.mp4"
    process_video(video_path, threshold=1250.0)

# link quality score to PIV result
# table of quality score and accuracy of PIV

# review: team is making good progress, in correct direction. great job.

