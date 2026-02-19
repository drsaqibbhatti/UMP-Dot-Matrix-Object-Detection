import os
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader
from utils.ObjDatasetSanity import ObjDatasetSanity  # Adjust path as needed
from torchvision.transforms import transforms
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

# Constants
dataset_path = "D:/hvs/Hyvsion_Projects/UMP/DotMatrix_Data/UMP/25EA_Done_250605/Inner_LT"
output_dir = "D:/hvs/Hyvsion_Projects/UMP/DotMatrix_Data/UMP_Overlay/Inner_LT/Inner_LT_NoAug"
os.makedirs(output_dir, exist_ok=True)

# Dataset & DataLoader
transform = transforms.Compose([
    transforms.ToTensor(),
])
transformAug = transforms.Compose([
    # transforms.ColorJitter(brightness=(0.5,1.2), contrast=(0.6,1.1)),
    # transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.3),
    # transforms.RandomAutocontrast(p=0.2),
    # transforms.RandomGrayscale(p=0.2),
    # transforms.RandomApply([
    #     transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.4))
    # ], p=0.3),
    transforms.ToTensor(),
])

imageHeight = 320
imageWidth = 480
dataset = ObjDatasetSanity(root=dataset_path, img_size=(imageWidth, imageHeight), useHFlip=True, useVFlip=True, transform=transformAug)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Color and label mapping
colors = [(0, 255, 0), (0, 0, 255)]  # one color per class
label_map = {0: "OneDot", 1: "TwoDot"}

# Visualization loop
for img_tensor, target, filename in dataloader:
    # Convert to HWC format and uint8 (OpenCV expects this)
    img_np = img_tensor[0].permute(1, 2, 0).numpy()  # [C, H, W] -> [H, W, C]
    img_np = (img_np * 255).astype(np.uint8)
    img_np = img_np[:, :, ::-1].copy()  # RGB -> BGR for OpenCV

    h, w, _ = img_np.shape
    boxes = target['boxes'][0]
    labels = target['labels'][0]

    for box, label in zip(boxes, labels):
        cx, cy, bw, bh = box
        cx *= w
        cy *= h
        bw *= w
        bh *= h

        x1 = int(cx - bw / 2)
        y1 = int(cy - bh / 2)
        x2 = int(cx + bw / 2)
        y2 = int(cy + bh / 2)

        cv2.rectangle(img_np, (x1, y1), (x2, y2), colors[label.item()], 2)
        cv2.putText(img_np, label_map[label.item()], (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[label.item()], 1)

    out_path = os.path.join(output_dir, filename[0])
    cv2.imwrite(out_path, img_np)


print("Overlay images saved.")




# Define directories for multiple images (grayscale comparison)
train_dir = output_dir
test_dir = "D:/hvs/Hyvsion_Projects/UMP/DotMatrix_Data/Test_data/Original_Test/AOI_TestSet_100EA/Outer_1"    # Folder with test images

def load_grayscale_images_from_folder(folder_path, limit=10):
    image_paths = sorted(glob(os.path.join(folder_path, "*.jpg")))[:limit]
    images = [np.array(Image.open(img_path).convert("L")) for img_path in image_paths]
    return images

def average_histogram(images):
    hist_sum = np.zeros((256,), dtype=np.float32)
    for img in images:
        hist = cv2.calcHist([img.astype(np.uint8)], [0], None, [256], [0, 256]).flatten()
        hist_sum += hist
    return hist_sum / len(images)

# Load up to 10 images from each folder
train_images = load_grayscale_images_from_folder(train_dir)
test_images = load_grayscale_images_from_folder(test_dir)

# Compute average histograms
avg_train_hist = average_histogram(train_images)
avg_test_hist = average_histogram(test_images)

# Plot comparison
plt.figure(figsize=(10, 5))
plt.plot(avg_train_hist, label="Average Training Histogram", color="green")
plt.plot(avg_test_hist, label="Average Test Histogram", color="red")
plt.title("Average Grayscale Histogram: Train vs Test")
plt.xlabel("Pixel Intensity (0=black, 255=white)")
plt.ylabel("Average Pixel Count")
plt.legend()
plt.grid(True)
plt.savefig(f"{output_dir}/histogram_comparison.png") 
plt.tight_layout()
plt.show()
