import os
import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from model.ObjModel import ObjLarge
from utils.util import non_max_suppression, visualize_predictions
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torchvision.transforms.functional as F
import numpy as np
import os

def Saving_Overlays(image, gt_boxes=None, pred_boxes=None, gt_labels=None, pred_labels=None,
                    pred_scores=None, save_path=None, label_map=None):
    """
    image: Tensor [3, H, W]
    gt_boxes: list of [x1, y1, x2, y2]
    pred_boxes: list of [x1, y1, x2, y2]
    pred_labels: list of class indices
    pred_scores: list of confidence scores
    save_path: optional path to save the visualized image
    label_map: dict, optional mapping from class index to label name
    """

    # Convert tensor to PIL image
    image = F.to_pil_image(image)
    width, height = image.size

    # Force exact pixel size: width = inches * dpi
    fig, ax = plt.subplots(figsize=(width / 100, height / 100), dpi=100)
    ax.imshow(image)
    ax.axis('off')

    # Ground truth boxes in green
    if gt_boxes is not None and len(gt_boxes) > 0:
        for i, box in enumerate(gt_boxes):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor='g', facecolor='none')
            ax.add_patch(rect)
            if gt_labels is not None:
                label_text = label_map[gt_labels[i]] if label_map else str(gt_labels[i])
                ax.text(x1, y1 - 5, f"GT: {label_text}",
                        color='green', fontsize=8, backgroundcolor='black')

    # Predicted boxes in red
    if pred_boxes is not None and len(pred_boxes) > 0:
        for i, box in enumerate(pred_boxes):
            x1, y1, x2, y2 = box
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            if pred_labels is not None and pred_scores is not None:
                label_text = label_map[pred_labels[i]] if label_map else str(pred_labels[i])
                score = pred_scores[i]
                ax.text(x1, y1 - 5, f"{label_text}: {score:.2f}",
                        color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))

    # Save or show
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()


# Settings
image_folder = "D:/hvs/Hyvsion_Projects/UMP/DotMatrix_Data/UMP/DataUpto_250613/Outer_2/validation/250611"  # Folder with input images
model_path = "D:/hvs/Hyvsion_Projects/UMP/UMP_git/python/Trained_models/HvsLarge/Outer_2/run_1/Outer_2_H800W320_2025-06-16_E98_iou_0.9783.pth"  # Path to the trained model
output_folder = "D:/hvs/Hyvsion_Projects/UMP/DotMatrix_Data/UMP_Test_overlays/DataUpto_250613/Outer_2/validation/250611/run_1"  # Folder to save overlay images
imageHeight = 800
imageWidth = 320
img_size = (imageHeight,imageWidth)  

os.makedirs(output_folder, exist_ok=True)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
model = ObjLarge(num_classes=2).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
])

# Loop over images
for filename in os.listdir(image_folder):
    if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue

    image_path = os.path.join(image_folder, filename)
    image_pil = Image.open(image_path).convert("RGB")
    image_tensor = transform(image_pil).unsqueeze(0).to(device)  # Add batch dimension

    # Get prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        predictions = non_max_suppression(outputs, conf_threshold=0.5, iou_threshold=0.5)

    # Convert to numpy for visualization
    image_np = np.array(image_pil.resize(img_size))

    # Overlay predictions
    if predictions[0] is not None:
        pred_boxes = predictions[0][:, :4].cpu().numpy()
        pred_scores = predictions[0][:, 4].cpu().numpy()
        pred_labels = predictions[0][:, 5].cpu().numpy().astype(int)

        Saving_Overlays(
            image=image_tensor[0].cpu(),  # Tensor of shape [3, H, W]
            gt_boxes=None,  # No GT during inference
            pred_boxes=pred_boxes,
            gt_labels=None,
            pred_labels=pred_labels,
            pred_scores=pred_scores,
            save_path=os.path.join(output_folder, filename)
        )

    else:
        print(f"No predictions for {filename}")
