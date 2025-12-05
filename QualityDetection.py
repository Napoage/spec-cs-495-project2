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
CNN_RESNET_MODEL_PATH = "../models/raindrop_classifier.pth"
XGB_QUALITY_MODEL_PATH = "../models/xgb_quality_ACS.json"
XGB_QUALITY_SCALER_PATH = "../models/scaler_ACS.joblib"
RAINDROP_VARIANTS_DIR = "RaindropVariants"
DEVICE = torch.device("cpu")

# Load ResNet18 raindrop classifier
resnet_model = models.resnet18(weights=None)
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(num_ftrs, 2)
)
state_dict = torch.load(CNN_RESNET_MODEL_PATH, map_location=DEVICE)
resnet_model.load_state_dict(state_dict)
resnet_model.to(DEVICE)
resnet_model.eval()

# Load XGB quality model
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(XGB_QUALITY_MODEL_PATH)
xgb_scaler = joblib.load(XGB_QUALITY_SCALER_PATH)

# Transform for CNN
transform = transforms.Compose([
    transforms.Resize((224, 224), antialias=True),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Feature extractor (same as before)
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
    return [sharpness, glare, contrast, ent, edge_density, mean_brightness,
            saturation, clipped_black_ratio, clipped_white_ratio]


# returns score for every frame
def raindropDetection(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)
    img_tensor = transform(frame_pil).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = resnet_model(img_tensor)
        probs = torch.softmax(output, dim=1)
        raindrop_prob = probs[0, 1].item()
        clean_prob = probs[0, 0].item()

    # If CNN says raindrop -> score reflects that
    if raindrop_prob > 0.5:
        score = (1 - raindrop_prob)  # low score for raindrops
        return {
            "type": "raindrop",
            "score": float(score),
            "clean_prob": float(clean_prob),
            "raindrop_prob": float(raindrop_prob)
        }

    # No raindrop: evaluate quality via XGB
    features = extract_features(frame)
    feature_names = [
        "sharpness","glare","contrast","entropy","edge_density",
        "mean_brightness","saturation","clipped_black_ratio","clipped_white_ratio"
    ]

    features_df = pd.DataFrame([features], columns=feature_names)
    scaled_features = xgb_scaler.transform(features_df)

    quality_prob = xgb_model.predict_proba(scaled_features)[0][1]  # P(good)

    # combined frame score
    frame_score = clean_prob * quality_prob

    return {
        "type": "quality",
        "score": float(frame_score),
        "clean_prob": float(clean_prob),
        "quality_prob": float(quality_prob)
    }


def passVideoForTesting(video, FRAME_SKIP=10, REJECT_THRESHOLD=0.5):
    cap = cv2.VideoCapture(video)
    frame_idx = 0
    stats = {
        "passed_quality": 0,
        "failed_quality": 0,
        "raindrop_detected": 0,
        "frame_scores": []
    }

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % FRAME_SKIP == 0:
            result = raindropDetection(frame)
            stats["frame_scores"].append(result["score"])

            if result["type"] == "quality":
                if result["score"] >= 0.5:
                    stats["passed_quality"] += 1
                else:
                    stats["failed_quality"] += 1
            else:
                stats["raindrop_detected"] += 1

        frame_idx += 1

    cap.release()

    score = results(stats["passed_quality"], stats["failed_quality"],
            stats["raindrop_detected"], stats, REJECT_THRESHOLD)
    
    return score

def syntheticRaindropAnalysis(frame_id, FRAME_SKIP=2, REJECT_THRESHOLD=0.5):
    frame_idx = 0
    stats = {
        "passed_quality": 0,
        "failed_quality": 0,
        "raindrop_detected": 0,
        "frame_scores": []
    }

    frames = sorted([
        os.path.join(RAINDROP_VARIANTS_DIR, f)
        for f in os.listdir(RAINDROP_VARIANTS_DIR)
        if f.lower().endswith((".jpg", ".png", ".jpeg")) and frame_id.lower() in f.lower()
    ])

    print(f"{len(frames)} frames found for videoID = {frame_id}")

    for frame_file in frames:
        opened_frame = cv2.imread(frame_file)
        if opened_frame is None:
            frame_idx += 1
            continue

        if frame_idx % FRAME_SKIP == 0:
            result = raindropDetection(opened_frame)
            stats["frame_scores"].append(result["score"])

            if result["type"] == "quality":
                if result["score"] >= 0.5:
                    stats["passed_quality"] += 1
                else:
                    stats["failed_quality"] += 1
            else:
                stats["raindrop_detected"] += 1

        frame_idx += 1

    score = results(stats["passed_quality"], stats["failed_quality"],
            stats["raindrop_detected"], stats, REJECT_THRESHOLD)
    
    return score

def results(passed, failed, raindrops, stats, REJECT_THRESHOLD):
    total = passed + failed + raindrops
    if total == 0:
        print("No frames processed!")
        return

    drop_detected = False
    # average frame score
    avg_score = sum(stats["frame_scores"]) / len(stats["frame_scores"])

    # raindrops just set to arbitrarily low value (5)
    if ((failed + raindrops) / total) >= REJECT_THRESHOLD or raindrops >= 5:
        print(f"Video is rejected due to lens obstruction or degraded video quality - Quality Insufficient ({((failed + raindrops) / total) * 100:.2f}%)")
        if raindrops >= 5:
            drop_detected = True
    else:
        print(f"Video Passed - Quality Sufficient ({(passed / total) * 100:.2f}%)")

    print(stats["frame_scores"])
    print(f"\nAverage Frame Quality Score: {avg_score:.3f}")
    
    return avg_score, drop_detected


def main():
    video_path = "ACS.MP4"
    frame_id = "ACS"
    frame_skip = 10
    reject_threshold = 0.5
    drop = False

    final_score, drop = syntheticRaindropAnalysis(frame_id, REJECT_THRESHOLD=reject_threshold)
    #final_score = passVideoForTesting(video_path, FRAME_SKIP=frame_skip, REJECT_THRESHOLD=reject_threshold)

if __name__ == "__main__":
    main()
