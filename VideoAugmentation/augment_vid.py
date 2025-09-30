import cv2
import os

orig_filename = input("Enter the name of the video: ")
base_name = os.path.splitext(orig_filename)[0]

video = cv2.VideoCapture(orig_filename)
if not video.isOpened():
    print("Error: Could not open video.")
    exit()

fps = int(video.get(cv2.CAP_PROP_FPS))
width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

augmentations = []

while True:
    choice = input(
        "OPTIONS:\n"
        "1 - BLUR IMAGE\n"
        "2 - LIGHTEN IMAGE\n"
        "3 - DARKEN IMAGE\n"
        "4 - DONE\n"
    )

    if choice in ["1", "2", "3"]:
        while True:
            choice2 = input(
                "FULL OR HALF:\n"
                "1 - FULL\n"
                "2 - HALF\n"
            )
            if choice2 == "1":
                mode = "full"
                break
            elif choice2 == "2":
                mode = "half"
                break
            else:
                print("Invalid, try again.")

        if choice == "1":  # blur
            while True:
                try:
                    blur_strength = int(input("Enter blur intensity (odd number, e.g., 5-51): "))
                    if blur_strength % 2 == 1 and blur_strength > 0:
                        break
                    else:
                        print("Must be a positive odd number.")
                except ValueError:
                    print("Enter a valid integer.")
            augmentations.append(("blur", mode, blur_strength))

        elif choice == "2":  # lighten
            while True:
                try:
                    lighten_beta = int(input("Enter lightening value (0-100, higher = brighter): "))
                    if 0 <= lighten_beta <= 100:
                        break
                    else:
                        print("Enter a value between 0 and 100.")
                except ValueError:
                    print("Enter a valid integer.")
            augmentations.append(("lighten", mode, lighten_beta))

        elif choice == "3":  # darken
            while True:
                try:
                    darken_beta = int(input("Enter darkening value (-100 to 0, lower = darker): "))
                    if -100 <= darken_beta <= 0:
                        break
                    else:
                        print("Enter a value between -100 and 0.")
                except ValueError:
                    print("Enter a valid integer.")
            augmentations.append(("darken", mode, darken_beta))

        print(f"Added augmentation: {augmentations[-1]}")
    elif choice == "4":
        if not augmentations:
            print("No augmentations selected. Exiting.")
            video.release()
            exit()
        break
    else:
        print("Invalid option, try again.")

final_filename = base_name
for aug in augmentations:
    effect, mode, intensity = aug
    final_filename += f"_{effect}_{mode}"
final_filename += ".mp4"

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(final_filename, fourcc, fps, (width, height))

print(f"Processing video with {len(augmentations)} augmentation(s)...")

while True:
    ret, frame = video.read()
    if not ret:
        print("Processing complete...")
        break

    processed_frame = frame.copy()
    h = processed_frame.shape[0]

    for aug in augmentations:
        effect, mode, intensity = aug
        if effect == "blur":
            if mode == "full":
                processed_frame = cv2.GaussianBlur(processed_frame, (intensity, intensity), 0)
            else:
                bottom_half = processed_frame[h//2:, :]
                processed_frame[h//2:, :] = cv2.GaussianBlur(bottom_half, (intensity, intensity), 0)
        elif effect == "lighten":
            if mode == "full":
                processed_frame = cv2.convertScaleAbs(processed_frame, alpha=1.0, beta=intensity)
            else:
                top_half = processed_frame[:h//2, :]
                processed_frame[:h//2, :] = cv2.convertScaleAbs(top_half, alpha=1.0, beta=intensity)
        elif effect == "darken":
            if mode == "full":
                processed_frame = cv2.convertScaleAbs(processed_frame, alpha=0.7, beta=intensity)
            else:
                bottom_half = processed_frame[h//2:, :]
                processed_frame[h//2:, :] = cv2.convertScaleAbs(bottom_half, alpha=0.7, beta=intensity)

    out.write(processed_frame)

    frame_count = int(video.get(cv2.CAP_PROP_POS_FRAMES))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count % 100 == 0:
        progress = (frame_count / total_frames) * 100
        print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")

video.release()
out.release()
cv2.destroyAllWindows()
print(f"Video saved as: {final_filename}")
