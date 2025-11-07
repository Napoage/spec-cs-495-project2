import sys
sys.path.append("/home/DylanBrearley-CVGraduateProject/utils/ROLE/raindrop")

import os
import cv2
import pandas as pd
from tqdm import tqdm
from dropgenerator import generateDrops

INPUT_DIR = "../Frames"
OUTPUT_DIR = "../RaindropVariants"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# same as train/test variables

cfg = { 
    "minDrops": 8, "maxDrops": 30, "minR": 20, "maxR": 80, "return_label": False, "edge_darkratio": 1.8, "label_thres": 0.12
}

def main():
    
    images = [f for f in os.listdir(INPUT_DIR)
              if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    
    print(f"{len(images)} total frames found.")
    
    for img_file in tqdm(images):
        img_path = os.path.join(INPUT_DIR, img_file)
        rainy = generateDrops(img_path, cfg)
        name, _ = os.path.splitext(img_file)
        rainy_out = os.path.join(OUTPUT_DIR, f"{name}_raindrop.jpg")
        rainy.save(rainy_out)
        
        
if __name__ == "__main__":
    main()

