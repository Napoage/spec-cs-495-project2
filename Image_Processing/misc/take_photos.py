import cv2
import os
import time

'''
Takes photos for camera calibration. User presses ENTER to take a photo
and types exit to quit. The images are saved to the 
Image_Processing/camera_calibration/images_for_calibration directory.
'''

# Create the folder for saving images
folder_name = "images_for_calibration"
os.makedirs(folder_name, exist_ok=True)

# Initialize the camera
camera = cv2.VideoCapture(2)  # 2 is a gstreamer virtual camera

if not camera.isOpened():
    print("Error: Could not open the camera. Make sure gstreamer is running with 'sudo systemctl status gstreamer.service'")
    exit()

try:
    while True:
        print("\nPress ENTER to capture an image or type 'exit' to quit.")
        command = input().strip().lower()

        if command == "exit":
            print("Exiting...")
            break

        # Flush the camera buffer to ensure a fresh frame
        for _ in range(10):  # Adjust the number of frames to flush if needed
            camera.read()
        ret, frame = camera.read()

        if not ret:
            print("Error: Unable to capture frame.")
            continue

        # Save the captured image
        timestamp = int(time.time() * 1000)  # Use milliseconds for a unique filename
        image_filename = os.path.join(folder_name, f"image_{timestamp}.jpg")
        cv2.imwrite(image_filename, frame)
        print(f"Saved: {image_filename}")

except KeyboardInterrupt:
    print("\nProgram interrupted. Exiting...")

finally:
    # Release the camera and close OpenCV windows
    camera.release()
    cv2.destroyAllWindows()
