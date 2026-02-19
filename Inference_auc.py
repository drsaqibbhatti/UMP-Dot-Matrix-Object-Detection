import torch
import random
import numpy as np
from datetime import date
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from model.ObjModel import ObjLarge
from utils.util import ComputeLoss, EMA, pixel_iou_from_nms,iou_from_nms_auc, clip_gradients, setup_seed, compute_iou, load_checkpoint, visualize_predictions
from utils.ObjDataset import ObjDataset
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import base64
import sys
import io
from sklearn.metrics import precision_recall_curve, auc

# Parameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def match_predictions_to_gt(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, iou_thresh=0.5):
    matched_gt = set()
    tp = 0
    fp = 0

    for i, (p_box, p_label) in enumerate(zip(pred_boxes, pred_labels)):
        matched = False
        for j, (g_box, g_label) in enumerate(zip(gt_boxes, gt_labels)):
            if j in matched_gt:
                continue
            iou = compute_iou(p_box, g_box)
            if iou >= iou_thresh and p_label == g_label:
                matched = True
                matched_gt.add(j)
                break
        if matched:
            tp += 1
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn




def inference():
    # Device configuration
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    print(f"The device is: {device}")


    if USE_CUDA:
        torch.cuda.manual_seed_all(777)

    # Hyperparameters


    # Transformationshow 
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    val_path= "D:/hvs/Hyvsion_Projects/UMP/DotMatrix_Data/UMP/DataUpto_250613/Outer_1/validation"
    data_type= "Outer_2"
    base_dir = "D:/hvs/Hyvsion_Projects/UMP/UMP_git/python/Trained_models/HvsLarge/Outer_2/run_2"
    model_path = "D:/hvs/Hyvsion_Projects/UMP/UMP_git/python/Trained_models/HvsLarge/Outer_2/run_2/Outer_2_H800W320_2025-06-30_E0_iou_0.5302.pth"  # Path to the trained model



    if data_type == "Inner_LT":
        imageHeight, imageWidth = 320, 480
    elif data_type == "Inner_RT":
        imageHeight, imageWidth = 640, 320
    elif data_type == "Outer_1":
        imageHeight, imageWidth = 320, 320
    elif data_type == "Outer_2":
        imageHeight, imageWidth = 800, 320
    else:
        print(f"Unsupported data_type: {data_type}")
        
    
    imageHeight = imageHeight
    imageWidth = imageWidth
    
    

    if not val_path or not os.path.isdir(val_path):
        raise ValueError(f"Invalid validation dataset path: {val_path}")


    
    validDataset = ObjDataset(
                            root=val_path,
                            transform=transform,
                            img_size=(imageWidth, imageHeight),
                            normalize_boxes=True)
        

    validLoader = DataLoader(validDataset, batch_size=1, shuffle=True, drop_last=False, collate_fn=lambda x: tuple(zip(*x)))
    ID_TO_LABEL = {
        "onedot": 0,
        "twodot": 1
    }
    
    LABEL_NAMES = {v: k for k, v in ID_TO_LABEL.items()}
    

    print(f"Number of validation samples: {len(validDataset)}")


    # Initialize model
    num_classes = 2
    # Load model
    model = ObjLarge(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
        







    thresholds = np.arange(0.09, 1.0, 0.1)
    results = []  # Store (threshold, precision, recall, f1)
    per_class_tp_dict = {}
    per_class_fp_dict = {}
    per_class_fn_dict = {}
    for obj_th in thresholds:
        total_tp = total_fp = total_fn = total_iou = 0.0

        with torch.no_grad():
            for i, (images_val, targets_val) in enumerate(validLoader):
                # Prepare images and targets
                images_val = torch.stack([img.to(device) for img in images_val], dim=0)
                targets_val = [{k: v.to(device).float() for k, v in t.items()} for t in targets_val]

                _, _, height, width = images_val.shape  # Get the actual image dimensions after resizing

                    
                # Get model predictions
                outputs = model(images_val)


                mean_iou, per_class_totals, per_class_counts, per_class_tp, per_class_fp, per_class_fn = iou_from_nms_auc(
                    outputs=outputs,
                    targets=targets_val,
                    width=width,
                    height=height,
                    object_score_th=0.05,
                    visualize=False
                )

                total_iou += mean_iou
                
                per_class_tp_dict[obj_th] = per_class_tp
                per_class_fp_dict[obj_th] = per_class_fp
                per_class_fn_dict[obj_th] = per_class_fn


                total_tp = sum(per_class_tp.values())
                total_fp = sum(per_class_fp.values())
                total_fn = sum(per_class_fn.values())

                print(f"per_class_tp: {per_class_tp}, per_class_fp: {per_class_fp}, per_class_fn: {per_class_fn}")
                print(f"total_tp: {total_tp}, total_fp: {total_fp}, total_fn: {total_fn}")

        overall_iou = total_iou / len(validLoader)

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall    = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0

        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)

        results.append((obj_th, overall_iou, precision, recall, f1_scores))
    # Convert results to DataFrame
        df = pd.DataFrame(results, columns=['threshold', 'IoU', 'precision', 'recall', 'f1_score'])
        csv_path = os.path.join(base_dir, "object_score_threshold_metrics.csv")
        df.to_csv(csv_path, index=False)

    
    
    thresholds, overall_iou, precision, recall, f1_scores = zip(*results)

    ####################################################################
    # Sort thresholds
    thresholds = sorted(per_class_tp_dict.keys())

    # Get all class IDs across all thresholds
    class_ids = sorted(set().union(*[d.keys() for d in per_class_tp_dict.values()]))
    class_labels = [str(cls_id) for cls_id in class_ids]

    # Layout for subplots
    n_cols = 3
    n_rows = int(np.ceil(len(thresholds) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 6, n_rows * 4), squeeze=False)

    for idx, threshold in enumerate(thresholds):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row][col]

        # Extract per-class counts at this threshold
        tp_vals = [per_class_tp_dict[threshold].get(cls, 0) for cls in class_ids]
        fp_vals = [per_class_fp_dict[threshold].get(cls, 0) for cls in class_ids]
        fn_vals = [per_class_fn_dict[threshold].get(cls, 0) for cls in class_ids]

        x = np.arange(len(class_ids))
        width = 0.25

        ax.bar(x - width, tp_vals, width=width, label='TP', color='green')
        ax.bar(x,         fp_vals, width=width, label='FP', color='red')
        ax.bar(x + width, fn_vals, width=width, label='FN', color='orange')

        ax.set_xticks(x)
        ax.set_xticklabels(class_labels, rotation=45)
        ax.set_title(f"Threshold: {threshold:.2f}")
        ax.set_ylabel("Count")

        if row == 0 and col == 0:
            ax.legend()

    # Remove unused subplots if any
    for idx in range(len(thresholds), n_rows * n_cols):
        fig.delaxes(axes[idx // n_cols][idx % n_cols])

    plt.tight_layout()
    plt.show()
    plot_path = os.path.join(base_dir, "per_class_histograms_across_thresholds.png")
    fig.savefig(plot_path)
    print(f"Saved histogram subplot to: {plot_path}")



    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='o', linestyle='-')
    for i, th in enumerate(thresholds):
        plt.annotate(f"{th:.2f}", (recall[i], precision[i]), fontsize=8, alpha=0.7)

    plt.title("Precision-Recall Curve")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    inference()
