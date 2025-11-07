import os
import cv2
import torch
import torchvision.transforms as transforms
import xgboost as xgb
import numpy as np
from PIL import Image
from tqdm import tqdm
import joblib
from skimage import exposure
from scipy.stats import entropy
import torch.nn as nn
import torchvision.models as models
import pandas as pd

#paths
CNN_RESNET_MODEL_PATH = "Models/raindrop_classifier.pth"
XGB_QUALITY_MODEL_PATH = "Models/BiggerDataset/xgb_quality_ACS.json"
XGB_QUALITY_SCALER_PATH = "Models/BiggerDataset/scaler_ACS.joblib"
RAINDROP_VARIANTS_DIR = "RaindropVariants"
DEVICE = torch.device("cpu") # keep cpu because this needs to run on pi

# resent model loading (resnet-18 CNN)
resnet_model = models.resnet18(weights=None)
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = nn.Sequential(
    nn.Dropout(0.4), # no effect in testing, dont need dropout but it throws an error when it is not included, this is because it needs to match the model's layer structure
    nn.Linear(num_ftrs, 2) # 0 or 1 for raindrop prediction
)
state_dict = torch.load(CNN_RESNET_MODEL_PATH, map_location=DEVICE)
resnet_model.load_state_dict(state_dict)
resnet_model.to(DEVICE)
resnet_model.eval()

# xgb model
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(XGB_QUALITY_MODEL_PATH)
xgb_scaler = joblib.load(XGB_QUALITY_SCALER_PATH)

# match transform from training (convert to pytorch tensor, resize, normalize colors)
transform = transforms.Compose([
    transforms.Resize((224, 224), antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# just extracts the features (same as training)
def extract_features(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    glare = np.mean(gray > 240)
    contrast = gray.std()
    hist, _ = np.histogram(gray, bins=256, range=(0, 255), density=True)
    ent = entropy(hist + 1e-7)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.mean(edges > 0)
    mean_brightness = gray.mean()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].mean()
    clipped_black_ratio = np.mean(gray < 10)
    clipped_white_ratio = np.mean(gray > 245)
    return [sharpness, glare, contrast, ent, edge_density, mean_brightness, saturation, clipped_black_ratio, clipped_white_ratio]

def raindropDetection(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # for pytorch (needs RGB)
    frame_pil = Image.fromarray(frame_rgb)
    img_tensor = transform(frame_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad(): # predict with CNN
        output = resnet_model(img_tensor)
        raindrop_pred = torch.argmax(output, dim=1).item()

    if raindrop_pred == 1: # immediately return -1 if raindrop is detected on lens
        return -1

    features = extract_features(frame) # otherwise extract features and send through XGB model

    feature_names = [
        "sharpness",
        "glare",
        "contrast",
        "entropy",
        "edge_density",
        "mean_brightness",
        "saturation",
        "clipped_black_ratio", 
        "clipped_white_ratio"
    ]

    features_df = pd.DataFrame([features], columns=feature_names)

    scaled_features = xgb_scaler.transform(features_df)
    quality_pred = xgb_model.predict(scaled_features)[0]
    return 1 if quality_pred == 1 else 0

def passVideoForTesting(video, FRAME_SKIP=10, REJECT_THRESHOLD=0.5):
    cap = cv2.VideoCapture(video)
    total_frames = 0
    frame_idx = 0
    stats = {"passed_quality": 0, "failed_quality": 0, "raindrop_detected": 0}
    
    #uncomment these next lines for progress bar

    #frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #pbar = tqdm(total=frame_count // FRAME_SKIP, desc="Processing frames")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_SKIP == 0:
            result = raindropDetection(frame)
            if result == 1:
                stats["passed_quality"] += 1
            elif result == 0:
                stats["failed_quality"] += 1
            elif result == -1:
                stats["raindrop_detected"] += 1
            total_frames += 1
            #pbar.update(1)

        frame_idx += 1

    cap.release()
    #pbar.close()

    results(stats["passed_quality"], stats["failed_quality"], stats["raindrop_detected"], stats, REJECT_THRESHOLD)
    
def syntheticRaindropAnalysis(frame_id, FRAME_SKIP=2, REJECT_THRESHOLD=0.5):
    
    frame_idx = 0
    total_frames = 0
    stats = {"passed_quality": 0, "failed_quality": 0, "raindrop_detected": 0}
    
    frames = sorted([
        os.path.join(RAINDROP_VARIANTS_DIR, f)
        for f in os.listdir(RAINDROP_VARIANTS_DIR)
        if f.lower().endswith((".jpg", ".png", ".jpeg")) and frame_id.lower() in f.lower()
    ])
    
    print(f"{len(frames)} found with videoID = {frame_id}")
    
    for frame in frames:
        opened_frame = cv2.imread(frame)
        if opened_frame is None:
            frame_idx += 1
            continue
        if frame_idx % FRAME_SKIP == 0:
            result = raindropDetection(opened_frame)
            if result == 1:
                stats["passed_quality"] += 1
            elif result == 0:
                stats["failed_quality"] += 1
            elif result == -1:
                stats["raindrop_detected"] += 1
            total_frames += 1
        frame_idx += 1
            
    results(stats["passed_quality"], stats["failed_quality"], stats["raindrop_detected"], stats, REJECT_THRESHOLD)

def results(passed, failed, raindrops, stats, REJECT_THRESHOLD):
    
    total = passed + failed + raindrops

    if total == 0:
        print("No frames were processed!")
        return

    if raindrops / total >= REJECT_THRESHOLD / 2:
        print(f"Video Rejected - Raindrops detected on lense: {(raindrops / total) * 100:.2f}%")
    elif passed / total >= REJECT_THRESHOLD:
        print(f"Video Passed - Video Quality Sufficient: {(passed / total) * 100:.2f}%")
    else:
        print(f"Video Rejected - Bad Frames: {(failed / total) * 100:.2f}%")
        
    print(stats)
    
def main():
    video_path = "Videos/ACS.MP4"
    frame_id = "CRG"
    frame_skip = 10
    reject_threshold = 0.5
    syntheticRaindropAnalysis(frame_id, REJECT_THRESHOLD=reject_threshold)
    #passVideoForTesting(video_path, FRAME_SKIP=frame_skip, REJECT_THRESHOLD=reject_threshold)

if __name__ == "__main__":
    main()
