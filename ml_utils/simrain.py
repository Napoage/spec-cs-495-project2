import sys
sys.path.append("/home/DylanBrearley-CVGraduateProject/utils/ROLE/raindrop")

import os
import cv2
import pandas as pd
from tqdm import tqdm
from dropgenerator import generateDrops
from sklearn.model_selection import train_test_split

# drop generator is a github repo that I found that programatically simulates raindrops, i updated it to work with python 3 because the last push was from 2019
# https://github.com/ricky40403/ROLE/blob/master/raindrop/dropgenerator.py

INPUT_DIR = "../datasets/coco_sample/data/train2017/train2017/"
OUTPUT_DIR = "../Data/RaindropDataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)
DATA_CSV = os.path.join(OUTPUT_DIR, "metadata.csv")

cfg = {
    "minDrops": 8, "maxDrops": 30, "minR": 20, "maxR": 80, "return_label": False, "edge_darkratio": 1.8, "label_thres": 0.12
}

def main():
    csv = []

    images = [f for f in os.listdir(INPUT_DIR)
              if f.lower().endswith((".jpg", ".png", ".jpeg"))]

    clean_imgs, rain_imgs = train_test_split(images, test_size=0.5, random_state=42)

    print(f"{len(clean_imgs)} CLEAN and {len(rain_imgs)} RAINDROP.")
# clean
    for img_file in tqdm(clean_imgs):
        img_path = os.path.join(INPUT_DIR, img_file)
        clean = cv2.imread(img_path)
        if clean is None:
            continue
        name, _ = os.path.splitext(img_file)
        clean_out = os.path.join(OUTPUT_DIR, f"{name}_clean.jpg")
        cv2.imwrite(clean_out, clean)
        csv.append([clean_out, 0])
# raindrop
    for img_file in tqdm(rain_imgs):
        img_path = os.path.join(INPUT_DIR, img_file)
        rainy = generateDrops(img_path, cfg)
        name, _ = os.path.splitext(img_file)
        rainy_out = os.path.join(OUTPUT_DIR, f"{name}_raindrop.jpg")
        rainy.save(rainy_out)
        csv.append([rainy_out, 1])
# save csv
    df = pd.DataFrame(csv, columns=["image_path", "label"])
    df.to_csv(DATA_CSV, index=False)

if __name__ == "__main__":
    main()
