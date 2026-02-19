import torch
import random
import numpy as np
from datetime import date
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from model.ObjModel import ObjLarge
from utils.util import ComputeLoss, EMA, non_max_suppression, pixel_iou_from_nms,iou_from_nms_auc, clip_gradients, setup_seed, compute_iou, load_checkpoint, visualize_predictions
from utils.ObjDatasetV2 import ObjDatasetV2
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import base64
import sys
import io
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from collections import defaultdict
import logging
from torch.utils.data import ConcatDataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def flush_stdout():
    sys.stdout.flush()


def run_inference(val_path, model_dir_path=None, imageHeight=int(320), imageWidth=int(320), data_type="default", auc_path=None):
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
    

    if not val_path or not all(os.path.isdir(p) for p in val_path):
        raise ValueError(f"Invalid one or more validation dataset paths: {val_path}")


    
    # val_datasets = []
    # validDataset = None
    # validLoader = None

    val_datasets = [
        ObjDatasetV2(root=q, transform=transform, img_size=(imageWidth, imageHeight), normalize_boxes=True)
        for q in val_path
    ]
    validDataset= ConcatDataset(val_datasets)
    validLoader = DataLoader(validDataset, batch_size=1, shuffle=True, drop_last=False, collate_fn=lambda x: tuple(zip(*x)))
        

    # ID_TO_LABEL = {
    #     "OneDot": 0,
    #     "TwoDot": 1,
    #     "Ignore": 2,
    # }
    
    # LABEL_NAMES = {v: k for k, v in ID_TO_LABEL.items()}
    

    # print(f"Number of validation samples: {len(validDataset)}")


    # Initialize model
    num_classes = 3
    # Load model
    model = ObjLarge(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_dir_path, map_location=device))
    model.eval()
        







    y_true_all = []
    y_scores_all = []
    per_class_tp = defaultdict(int)
    per_class_fp = defaultdict(int)
    per_class_fn = defaultdict(int)
    with torch.no_grad():
        for i, (images_val, targets_val) in enumerate(tqdm(validLoader, desc="Running inference", unit="image")):
            images_val = torch.stack([img.to(device) for img in images_val])
            targets_val = [{k: v.to(device).float() for k, v in t.items()} for t in targets_val]

            _, _, height, width = images_val.shape
            outputs = model(images_val)
            for target in targets_val:
                boxes = target['boxes']
                x_center, y_center, box_width, box_height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
                x_center *= width
                y_center *= height
                box_width *= width
                box_height *= height
                x_min = x_center - (box_width / 2)
                y_min = y_center - (box_height / 2)
                x_max = x_center + (box_width / 2)
                y_max = y_center + (box_height / 2)
                target['boxes'] = torch.stack([x_min, y_min, x_max, y_max], dim=1)
                
            outputs_nms = non_max_suppression(outputs, conf_threshold=0.01, iou_threshold=0.5)
            


            for idx, output in enumerate(outputs_nms):
                if output is None or output.size(0) == 0:
                    continue
                

                pred_boxes = output[:, :4]
                pred_scores = output[:, 4]
                pred_classes = output[:, 5].long()

                gt_boxes = targets_val[idx]['boxes']
                gt_labels = targets_val[idx]['labels'].long()
                matched_gt = torch.zeros(gt_boxes.size(0), dtype=torch.bool, device=device)
                matched_pred = torch.zeros(pred_boxes.size(0), dtype=torch.bool, device=device)

                for i in range(pred_boxes.size(0)):
                    box = pred_boxes[i]
                    score = pred_scores[i].item()
                    label = pred_classes[i].item()

                    class_mask = gt_labels == label
                    gt_class_boxes = gt_boxes[class_mask]

                    if gt_class_boxes.size(0) == 0 or score < 0.5:
                        y_true_all.append(0)
                        y_scores_all.append(score)
                        per_class_fp[label] += 1
                        continue

                    ious = compute_iou(box.unsqueeze(0), gt_class_boxes)
                    max_iou, max_idx = ious.max(dim=1)

                    if max_iou.item() >= 0.5:
                        global_idx = torch.nonzero(class_mask)[max_idx.item()]
                        if not matched_gt[global_idx] and not matched_pred[i]:
                            matched_gt[global_idx] = True
                            matched_pred[i] = True
                            y_true_all.append(1)
                            per_class_tp[label] += 1
                        else:
                            y_true_all.append(0)
                            per_class_fp[label] += 1
                    else:
                        y_true_all.append(0)
                        per_class_fp[label] += 1

                    y_scores_all.append(score)

                # Count FN per unmatched GT
                for i, (g_label, is_matched) in enumerate(zip(gt_labels.tolist(), matched_gt.tolist())):
                    if not is_matched:
                        per_class_fn[g_label] += 1



        ####################################################################
        if not y_true_all or not y_scores_all:
            logging.info(f"Predictions Not Found. No data to compute AUC.")
            flush_stdout()
            plt.figure()
            plt.text(0.5, 0.5, 'No Data Available', fontsize=14, ha='center')
            plt.axis('off')
            os.makedirs(auc_path, exist_ok=True)
            plt.savefig(os.path.join(auc_path, "auc_plots.png"))
            return

        # Compute PR curve
        y_true_np = np.array(y_true_all)
        y_scores_np = np.array(y_scores_all)

        precision, recall, thresholds = precision_recall_curve(y_true_np, y_scores_np)
        ap = average_precision_score(y_true_np, y_scores_np)
        
        # Compute F1 scores for each threshold
        f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
        thresholds_full = list(thresholds) + [1.0]  # match length

        # Calculate TPR and FPR (sorted by score descending)
        sorted_indices = np.argsort(-y_scores_np)
        y_true_sorted = y_true_np[sorted_indices]

        tp_cumsum = np.cumsum(y_true_sorted)
        fp_cumsum = np.cumsum(1 - y_true_sorted)

        total_positives = y_true_np.sum()
        total_negatives = len(y_true_np) - total_positives

        tpr = tp_cumsum / (total_positives + 1e-8)  # True Positive Rate
        fpr = fp_cumsum / (total_negatives + 1e-8)  # False Positive Rate

        plt.figure(figsize=(12, 8))

        # 1. TPR vs FPR
        plt.subplot(2, 3, 1)
        plt.plot(fpr, tpr, marker='.')
        plt.xlabel("False Positive Rate (FPR)")
        plt.ylabel("True Positive Rate (TPR)")
        plt.title("TPR vs FPR")
        plt.grid(True)

        # 2. Score vs F1
        plt.subplot(2, 3, 2)
        plt.plot(thresholds_full, f1_scores)
        plt.xlabel("Score Threshold")
        plt.ylabel("F1 Score")
        plt.title("Score vs F1 Score")
        plt.grid(True)
        # 3. Score vs Precision
        plt.subplot(2, 3, 3)
        plt.plot(thresholds_full, precision)
        plt.xlabel("Score Threshold")
        plt.ylabel("Precision")
        plt.title("Score vs Precision")
        plt.grid(True)
        # 4. Score vs Recall
        plt.subplot(2, 3, 4)
        plt.plot(thresholds_full, recall)
        plt.xlabel("Score Threshold")
        plt.ylabel("Recall")
        plt.title("Score vs Recall")
        plt.grid(True)
        plt.subplot(2, 3, 5)
        plt.plot(recall, precision, marker='.')
        plt.title(f'Precision-Recall Curve (AP = {ap:.4f})')
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(auc_path, "auc_plots.png"))



        # # Save raw data
        # df_pr = pd.DataFrame({
        #     "threshold": list(thresholds) + [1.0],
        #     "precision": precision,
        #     "recall": recall
        # })
        # df_pr.to_csv(os.path.join(base_dir, "pr_curve_data.csv"), index=False)


if __name__ == '__main__':
    run_inference()
