import cv2
import os

def extract_frames_large_gap(video_path, output_dir, frame_skip=10, max_frames=20):
    """
    Extract frames from video with larger time gaps for better PIV results

    Args:
        video_path: Path to input video
        output_dir: Directory to save frames
        max_frames: Maximum number of frames to extract
        frame_skip: Extract every nth frame (larger gap = more motion)
    """

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Clear existing frames
    for f in os.listdir(output_dir):
        if f.endswith('.png'):
            os.remove(os.path.join(output_dir, f))

    # Open video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    frame_count = 0
    extracted_count = 0

    print(f"Extracting frames from {video_path} with gap of {frame_skip} frames...")

    while True:
        ret, frame = cap.read()

        if not ret or extracted_count >= max_frames:
            break

        # Skip frames for larger temporal spacing
        if frame_count % frame_skip == 0:
            # Convert to grayscale for PIV
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Save frame
            frame_filename = f"{output_dir}/frame_{extracted_count:03d}.png"
            cv2.imwrite(frame_filename, gray_frame)
            extracted_count += 1

            print(f"Extracted frame {extracted_count} at video frame {frame_count}")

        frame_count += 1

    cap.release()
    print(f"Extracted {extracted_count} frames to {output_dir}")
    print(f"Time gap between frames: {frame_skip} video frames")

    return extracted_count

if __name__ == "__main__":
    print("Extracting frames with LARGER time gaps for better PIV...\n")

    # Extract with much larger gaps (every 10th frame instead of every 2nd)
    count1 = extract_frames_large_gap("Water Moving.mp4", "images", frame_skip=10, max_frames=15)

    # Also try the slow video with moderate gap
    count2 = extract_frames_large_gap("Water Moving Slow.mp4", "images_slow_gap", frame_skip=5, max_frames=15)

    print(f"\nExtraction complete!")
    print(f"Regular video with large gap: {count1} frames in 'images/'")
    print(f"Slow video with moderate gap: {count2} frames in 'images_slow_gap/'")
    print(f"\nNow frames have more time between them = more visible motion for PIV!")