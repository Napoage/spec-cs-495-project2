# Run the script:
Open a terminal and run:
python augment_vid.py or python3 augment_vid.py depending on your setup.

# Select your video:
The script will ask you to type the name of the video file you want to augment. Make sure it’s in the same folder as the script (or provide the full path).

# Choose augmentations:
Blur the video
Lighten the video
Darken the video
For each effect, you’ll choose whether to apply it to the full frame or half of the frame, and then specify the intensity.

# Processing:
The script reads the video frame by frame.
Each augmentation is applied to the chosen part of the frame.
The processed frames are written to a new video file.
Progress updates are shown every 100 frames.
Unfortunately, because frames are written frame by frame, the processing takes roughly the same amount of time as the video’s duration.

# Output:
When done, the new video is saved as an MP4 file.
The filename includes all the augmentations applied. For example, video_blur_full_lighten_half.mp4.

# You can now use the augmented video to test your goodness metric or any other analysis.