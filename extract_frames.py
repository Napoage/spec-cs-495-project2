import cv2
import os

def extract_frames_from_video(video_path, output_dir, max_frames=50, frame_skip=2):
    """
    Extract frames from video for PIV analysis

    Args:
        video_path: Path to input video
        output_dir: Directory to save frames
        max_frames: Maximum number of frames to extract
        frame_skip: Extract every nth frame (for temporal spacing)
    """

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Open video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    frame_count = 0
    extracted_count = 0

    print(f"Extracting frames from {video_path}...")

    while True:
        ret, frame = cap.read()

        if not ret or extracted_count >= max_frames:
            break

        # Skip frames for temporal spacing
        if frame_count % frame_skip == 0:
            # Convert to grayscale for PIV
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Save frame
            frame_filename = f"{output_dir}/frame_{extracted_count:03d}.png"
            cv2.imwrite(frame_filename, gray_frame)
            extracted_count += 1

            if extracted_count % 10 == 0:
                print(f"Extracted {extracted_count} frames...")

        frame_count += 1

    cap.release()
    print(f"Extracted {extracted_count} frames to {output_dir}")

    return extracted_count

if __name__ == "__main__":
    # Extract from both videos
    print("Extracting frames for PIV analysis...\n")

    # Extract from regular speed video
    count1 = extract_frames_from_video("Water Moving.mp4", "images", max_frames=30, frame_skip=2)

    # Extract from slow video to different folder for comparison
    count2 = extract_frames_from_video("Water Moving Slow.mp4", "images_slow", max_frames=30, frame_skip=2)

    print(f"\nExtraction complete!")
    print(f"Regular speed: {count1} frames in 'images/' directory")
    print(f"Slow speed: {count2} frames in 'images_slow/' directory")
    print(f"\nYou can now run PIV analysis on either set of frames.")