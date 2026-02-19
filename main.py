import torch
import random
import numpy as np
from datetime import date
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from model.ObjModel import ObjNano
from utils.util import ComputeLoss, ComputeIoU, EMA, clip_gradients, setup_seed, save_checkpoint, load_checkpoint, non_max_suppression, box_iou, visualize_predictions
from utils.ObjDataset import ObjDataset
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import base64
import sys
import io
# Parameters

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seed for reproducibility
setup_seed()

# Data transformation (no resizing, handled in dataset class)
transform = transforms.Compose([
    transforms.ToTensor()  # Convert image to PyTorch tensor
])




def train():
    # Device configuration
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    print(f"The device is: {device}")

    # Set random seeds for reproducibility
    random.seed(777)
    torch.manual_seed(777)
    if USE_CUDA:
        torch.cuda.manual_seed_all(777)

    # Hyperparameters


    # Transformationshow 
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load config
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.json"
    with open(config_path, "r", encoding='utf-8') as f:
        config = json.load(f)

    
    
    dataset_path = config.get("dataset_path")


    imageHeight = config.get("imageHeight")
    imageWidth = config.get("imageWidth")
    
    
    trainBatchSize = config.get("trainBatchSize")
    learningRate = config.get("learningRate")
    epochs = config.get("epochs")
    target_iou = 0.999
    
    if not dataset_path or not os.path.isdir(dataset_path):
        raise ValueError(f"Invalid dataset path: {dataset_path}")
    trainDataset = ObjDataset(
                                root=dataset_path,
                                transform=transform,
                                img_size=(imageWidth, imageHeight),
                                normalize_boxes=True,datatype='train')

    
    validDataset = ObjDataset(
                            root=dataset_path,
                            transform=transform,
                            img_size=(imageWidth, imageHeight),
                            normalize_boxes=True,datatype='val')
        
    trainLoader = DataLoader(trainDataset, batch_size=trainBatchSize, shuffle=True, drop_last=True, collate_fn=lambda x: tuple(zip(*x)))
    validLoader = DataLoader(validDataset, batch_size=1, shuffle=True, drop_last=False, collate_fn=lambda x: tuple(zip(*x)))
    ID_TO_LABEL = {
        "OneDot": 0,
        "TwoDot": 1
    }
    
    LABEL_NAMES = {v: k for k, v in ID_TO_LABEL.items()}
    
    print(f"Number of training samples: {len(trainDataset)}")
    print(f"Number of validation samples: {len(validDataset)}")


    # Initialize model
    num_classes = 2
    model = ObjNano(num_classes=num_classes)
    

    # # Multi-GPU support
    # if torch.cuda.device_count() > 1:
    #     print(f"Using {torch.cuda.device_count()} GPUs")
    #     model = DataParallel(model)

    model = model.to(device)

    # Loss and optimizer
    loss_fn = ComputeLoss(model, params={'cls': 0.5, 'box': 12.5, 'dfl': 2.5})
    ComputeIoUfn=ComputeIoU(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
    ema = EMA(model)
    amp_scaler = torch.cuda.amp.GradScaler() 
    # Track best IoU
    best_iou = 0.0
    best_model_path = None
    best_model_onnx_path = None
    best_model_dom_path = None

    # Directory to save model
    if getattr(sys, 'frozen', False):
        # If running from a bundled executable
        current_dir = os.path.dirname(sys.executable)
    else:
        # If running from a script
        current_dir = os.path.dirname(os.path.abspath(__file__))

   
    dataset_name = os.path.basename(os.path.normpath(dataset_path))
    Base_dir = os.path.join(current_dir, 'Trained_models',dataset_name)
    os.makedirs(Base_dir, exist_ok=True) 
    build_date = str(date.today())

    existing_runs = [d for d in os.listdir(Base_dir) if os.path.isdir(os.path.join(Base_dir, d))]
    run_number = len(existing_runs) + 1
    run_dir = os.path.join(Base_dir, f"run_{run_number}")
    os.makedirs(run_dir)
    
    csv_pathTrain = os.path.join(run_dir, "train_data_labels.csv")
    trainDataset.generate_csv(csv_pathTrain)
    csv_pathVal = os.path.join(run_dir, "Val_data_labels.csv")
    validDataset.generate_csv(csv_pathVal)
    metrics = []

    # Check if a checkpoint exists
    checkpoint_dir = os.path.join(run_dir, 'Checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    start_epoch = 0
    load_previous_checkpoint=True
    if load_previous_checkpoint:
        previous_run_number = run_number - 1
        previous_run_dir = os.path.join(Base_dir, f"run_{previous_run_number}")
        previous_checkpoint_dir = os.path.join(previous_run_dir, 'Checkpoints')

        if os.path.exists(previous_checkpoint_dir):
            checkpoint_files = sorted([f for f in os.listdir(previous_checkpoint_dir) if f.startswith('checkpoint_epoch_')])
            if checkpoint_files:
                last_checkpoint = checkpoint_files[-1]
                checkpoint_path = os.path.join(previous_checkpoint_dir, last_checkpoint)
                start_epoch = load_checkpoint(checkpoint_path, model, optimizer) + 1
                print(f"Loaded checkpoint from {previous_checkpoint_dir} at epoch {start_epoch}")
            else:
                print("No checkpoint files found in the previous run.")
        else:
            print("No previous run found to load a checkpoint from.")
    else:
        print("Starting from scratch, not loading any previous model.")
    saving_metrics = []
    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        train_ious = 0.0
        optimizer.zero_grad()

        with tqdm(total=len(trainLoader), desc=f"Epoch {epoch}/{epochs-1}", unit="batch") as pbar:
            for images, targets in trainLoader:
                images = torch.stack([img.to(device) for img in images], dim=0)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                
                # Forward pass with AMP
                with torch.cuda.amp.autocast():
                    outputs = model(images)

                    loss = loss_fn(outputs, targets)  # Calculate the loss

                # Backward pass with scaled gradients
                amp_scaler.scale(loss).backward()

                # Clip gradients and step optimizer
                amp_scaler.unscale_(optimizer)  # Unscale gradients before clipping
                clip_gradients(model, max_norm=1.0)
                amp_scaler.step(optimizer)
                amp_scaler.update()
                optimizer.zero_grad()  # Clear gradients

                ema.update(model)  # Update EMA

                train_loss += loss.item()
                

                with torch.no_grad():
                    pred_box_train, target_box_train = ComputeIoUfn(outputs, targets)
                    ious=ComputeIoU.iou(pred_box_train, target_box_train)
                    batch_mean_iou = ious.mean().item()
                    train_ious+= batch_mean_iou
                    
                    
                    
                pbar.update(1)
                pbar.set_postfix(loss=loss.item(), ious=batch_mean_iou)
        train_loss /= len(trainLoader)

        train_mean_iou =train_ious/ len(trainLoader)

                    
        print(f"Train Loss:{train_loss:.9f}, Train IoU: {train_mean_iou:.9f}")

        # Save checkpoint after each epoch
        checkpoint_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }
        save_checkpoint(checkpoint_state, model, checkpoint_dir, epoch)



        # Save best model based on IoU
        if train_mean_iou > best_iou:
            if best_model_path is not None:
                os.remove(best_model_path)
            best_iou = train_mean_iou
            best_model_path = os.path.join(run_dir, f"{dataset_name}_{build_date}_E{epoch}_iou_{best_iou:.4f}.pth")
            torch.save(ema.ema.state_dict(), best_model_path)
            # print(f"Saved best model at epoch {epoch} with train IoU : {best_iou:.4f}")


            if best_model_onnx_path is not None and os.path.exists(best_model_onnx_path):
                os.remove(best_model_onnx_path)

            if best_model_dom_path is not None and os.path.exists(best_model_dom_path):
                os.remove(best_model_dom_path)
                
            # Export to ONNX
            onnx_export_path = os.path.join(run_dir, f"{dataset_name}_{build_date}_E{epoch}_iou_{best_iou:.4f}.onnx")
            dom_export_path = os.path.join(run_dir, f"{dataset_name}_{build_date}_E{epoch}_iou_{best_iou:.4f}.dom")
            dummy_input = torch.randn(1, 3, imageHeight, imageWidth).to(device)

            # Use the EMA model for export
            ema.ema.eval()
            buffer = io.BytesIO()
            torch.onnx.export(ema.ema,
                            dummy_input,
                            buffer,
                            export_params=True,
                            opset_version=17,
                            do_constant_folding=True,
                            input_names=['input'],
                            output_names=['output'])
            buffer.seek(0)
            bytes_data = buffer.read()

            # Save ONNX to disk (optional)
            with open(onnx_export_path, "wb") as f:
                f.write(bytes_data)
            # print(f"Exported ONNX model to: {onnx_export_path} at epoch {epoch}")
            best_model_onnx_path = onnx_export_path

            

            base64Model = base64.b64encode(bytes_data).decode('ascii')

            # Create custom DSM JSON structure
            hvs_model_json = {
                "inputHeight": imageHeight,
                "inputWidth": imageWidth,
                "module": base64Model,
                "headType": "yolo8",
                "classNames": ["onedot", 'twodot']
            }

            # Save DSM format
            jsonFormatString = json.dumps(hvs_model_json)
            with open(dom_export_path, "w") as text_file:
                text_file.write(jsonFormatString)
            # print(f"Exported DOM model to: {dom_export_path} at epoch {epoch}")
            best_model_dom_path = dom_export_path
        
        # Early stopping based on IoU
        if train_mean_iou >= target_iou:
            print(f"Target IoU reached at epoch {epoch}. Stopping training.")
            break


        class_iou_totals = {}  # To track IoU for each class
        class_box_counts = {}  # To track the number of GT boxes per class


        total_iou = 0.0
        total_boxes = 0

        model.eval()
        with torch.no_grad():
            for i, (images_val, targets_val) in enumerate(validLoader):
                # Prepare images and targets
                images_val = torch.stack([img.to(device) for img in images_val], dim=0)
                targets_val = [{k: v.to(device).float() for k, v in t.items()} for t in targets_val]

                _, _, height, width = images_val.shape  # Get the actual image dimensions after resizing


                for target in targets_val:
                    boxes = target['boxes']
                    x_center, y_center, box_width, box_height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

                    # Convert to pixel values
                    x_center *= width
                    y_center *= height
                    box_width *= width
                    box_height *= height

                    # Convert to [x_min, y_min, x_max, y_max]
                    x_min = x_center - (box_width / 2)
                    y_min = y_center - (box_height / 2)
                    x_max = x_center + (box_width / 2)
                    y_max = y_center + (box_height / 2)

                    # Update target boxes with the converted values
                    target['boxes'] = torch.stack([x_min, y_min, x_max, y_max], dim=1)

                    
                ########### Filter out zero boxes from ground truth targets ###############################
                for target in targets_val:
                    gt_boxes = target['boxes']
                    valid_mask = (gt_boxes.sum(dim=1) > 0)  # Create mask for non-zero boxes
                    target['boxes'] = gt_boxes[valid_mask]  # Update target with valid boxes
                    target['labels'] = target['labels'][valid_mask]  # Update labels with valid boxes
                ##############################################################################################   
                    #print('target boxes in eval', target['boxes'])
                    
                # Get model predictions
                outputs = model(images_val)

                
                
                # Apply Non-Maximum Suppression (NMS)
                outputs = non_max_suppression(outputs, conf_threshold=0.001, iou_threshold=0.5)

                
                for j, output in enumerate(outputs):
                    if output is None or output.size(0) == 0:  # Skip if no predictions
                        continue

                    target_box = targets_val[j]['boxes']
                    target_class = targets_val[j]['labels'].long()
                    
                    predicted_scores = output[:, 4]
                    
                    # Predictions
                    predicted_boxes = output[:, :4]
                    predicted_classes = output[:, 5].long()
                    
                    # Clamp boxes
                    predicted_boxes[:, [0, 2]] = torch.clamp(predicted_boxes[:, [0, 2]], min=0, max=imageWidth)
                    predicted_boxes[:, [1, 3]] = torch.clamp(predicted_boxes[:, [1, 3]], min=0, max=imageHeight)

                    total_image_iou = 0
                    matched_boxes = 0


                    ############################### VISUALIZE: Prepare lists for visualization####################
                    vis_gt_boxes = []
                    vis_pred_boxes = []
                    vis_pred_labels = []
                    vis_pred_scores = []
                    ######################################################################################
                    
                    # Loop over each unique class in the ground truth
                    for cls in torch.unique(target_class):
                        # Filter ground truth and predictions for the current class
                        gt_mask = target_class == cls
                        pred_mask = predicted_classes == cls

                        gt_boxes_cls = target_box[gt_mask]
                        pred_boxes_cls = predicted_boxes[pred_mask]

                        # print("gt_boxes_cls",gt_boxes_cls)
                        # print("pred_boxes_cls",pred_boxes_cls)
                        
                        # Skip if no ground truth or predictions for this class
                        if gt_boxes_cls.size(0) == 0 or pred_boxes_cls.size(0) == 0:
                            continue

                        # Compute IoU for this class
                        iou = box_iou(gt_boxes_cls, pred_boxes_cls)
                        # print(f"IoU Matrix for Class {cls}:\n{iou}")
                        


                        ################### VISUALIZE: Collect the ground truth and corresponding best predictions ##########################
                        max_ious_per_gt, max_indices = torch.max(iou, dim=1)  # Max IoU per ground truth box
                        for gt_idx, pred_idx in enumerate(max_indices):
                            vis_gt_boxes.append(gt_boxes_cls[gt_idx].cpu().numpy())
                            vis_pred_boxes.append(pred_boxes_cls[pred_idx].cpu().numpy())
                            vis_pred_labels.append(cls.item())
                            vis_pred_scores.append(predicted_scores[pred_mask][pred_idx].cpu().item())
                        #############################################################################################################
                        
                        total_image_iou += max_ious_per_gt.sum().item()
                        matched_boxes += gt_boxes_cls.size(0)
                                                # Update per-class totals
                        if cls.item() not in class_iou_totals:
                            class_iou_totals[cls.item()] = 0.0
                            class_box_counts[cls.item()] = 0

                        class_iou_totals[cls.item()] += max_ious_per_gt.sum().item()
                        class_box_counts[cls.item()] += gt_boxes_cls.size(0)

                    # Update totals
                    total_iou += total_image_iou
                    total_boxes += matched_boxes


                    # Visualize the image after processing all classes
                    visualize_predictions(
                        image=images_val[j],  # Image tensor
                        gt_boxes=np.array(vis_gt_boxes),  # Ground truth boxes
                        pred_boxes=np.array(vis_pred_boxes),  # Predicted boxes
                        gt_labels=None,  # Ground truth labels are optional
                        pred_labels=vis_pred_labels,  # Predicted labels
                        pred_scores=vis_pred_scores  # Confidence scores
                    )
    

        overall_iou = total_iou / total_boxes if total_boxes > 0 else 0
        print(f"Validation IoU: {overall_iou}")

        per_class_iou = {}
        # Calculate and display per-class IoU
        print("\nPer-Class IoU:")
        for cls, iou_total in class_iou_totals.items():
            class_mean_iou = iou_total / class_box_counts[cls]
            class_name = LABEL_NAMES.get(cls, f"Class_{cls}")
            per_class_iou[f"{class_name}_IoU"] = round(class_mean_iou, 9)
            print(f"{class_name}_IoU: {class_mean_iou:.4f}")
        print("\n")

            


        saving_metrics.append({
            'Epoch': epoch,
            'TrainLoss': round(train_loss,4),
            'TrainIoU': round(train_mean_iou,9),
            'ValidIoU': round(overall_iou,9),
            **per_class_iou
        })
        # Save metrics every 10 epochs or at the last epoch
        metrics_df = pd.DataFrame(saving_metrics)
        csv_path = os.path.join(run_dir, f"training_metrics.csv")
        metrics_df.to_csv(csv_path, index=False)
        plot_metrics(metrics_df, run_dir, epoch)
            
    # Save final model
    final_model_path = os.path.join(run_dir, f"{build_date}_last_epoch_{epoch}_train_iou_{train_mean_iou:.4f}.pth")
    torch.save(ema.ema.state_dict(), final_model_path)
    print(f"Saved final model after epoch {epoch}")


def plot_metrics(metrics_df, run_dir, epoch):
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_df['Epoch'], metrics_df['TrainIoU'], label='TrainIoU')
    plt.plot(metrics_df['Epoch'], metrics_df['ValidIoU'], label='ValidIoU')

    for col in metrics_df.columns:
        if col.startswith("Class_"):  # Identify class IoU columns
            plt.plot(metrics_df['Epoch'], metrics_df[col], label=f"{col} IoU", linestyle='--')

    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.title(f'IoU (Up to Epoch {epoch})')
    plt.legend()
    plt.ylim(0, 1) 
    plt.grid(True)
    plt.savefig(os.path.join(run_dir, f'IoU.png'))
    plt.close()
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_df['Epoch'], metrics_df['TrainLoss'], label='TrainLoss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss (Up to Epoch {epoch})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(run_dir, f'Loss.png'))
    plt.close()

if __name__ == '__main__':
    train()
