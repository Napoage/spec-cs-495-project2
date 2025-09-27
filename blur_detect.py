import os
import cv2
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
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Break the loop if there are no more frames

        frame_count += 1

        # Resize the frame for faster processing (optional)
        frame = cv2.resize(frame, (200, 140))

        # Check if the frame is blurry
        _, blurry, variance = is_blurry(frame, threshold=threshold)

        print(
            f"Frame {frame_count} is {'blurry' if blurry else 'sharp'} (Variance: {variance:.2f})"
        )

        # Optional: Display the frame (press 'q' to quit)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


# Example usage
video_path = (
    r"Path/video.mp4"  # Replace with your video path
)
process_video(video_path, threshold=1250.0)