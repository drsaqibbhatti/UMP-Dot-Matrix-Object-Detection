import torch
import random
import numpy as np
from datetime import date
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from model.ObjModel import ObjLarge
from utils.util import ComputeLoss, EMA, pixel_iou_from_nms, clip_gradients, setup_seed, save_checkpoint, load_checkpoint, visualize_predictions
from utils.ObjDatasetV2 import ObjDatasetV2
import os
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import base64
import sys
import io
# Parameters
import logging
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.utils.data import ConcatDataset
# Set random seed for reproducibility
setup_seed()


def flush_stdout():
    sys.stdout.flush()

def run_train(train_path, val_path,result_model_dir_path=None, imageHeight=int(320), imageWidth=int(320), data_type="default", resume_training_path=None):
    # Device configuration
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda" if USE_CUDA else "cpu")
    logging.info(f"The device is: {device}")
    flush_stdout()

    # Set random seeds for reproducibility
    random.seed(777)
    torch.manual_seed(777)
    if USE_CUDA:
        torch.cuda.manual_seed_all(777)

    logging.info(f"Image Size: {imageHeight}x{imageWidth} (type: {type(imageHeight)}, {type(imageWidth)})")
    flush_stdout()
    # Hyperparameters

    # Transformationshow 
    transformAug = transforms.Compose([
        transforms.ColorJitter(brightness=(0.25, 1), contrast=(0.8, 1.2)),
        transforms.RandomAdjustSharpness(sharpness_factor=2.5, p=0.4),
        transforms.RandomAutocontrast(p=0.3),
        transforms.RandomGrayscale(p=0.05),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.4))
        ], p=0.05),
        transforms.ToTensor(),
    ])
    # Transformationshow 
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])


    # imageHeight = imageHeight
    # imageWidth = imageWidth
    epochs = 50
    
    trainBatchSize = 2
    learningRate = 0.001
    target_iou = 0.999
    
    if not train_path or not all(os.path.isdir(p) for p in train_path):
        raise ValueError(f"Invalid one or more train dataset paths: {train_path}")
    
    if not val_path or not all(os.path.isdir(p) for p in val_path):
        raise ValueError(f"Invalid one or more validation dataset paths: {val_path}")
    # trainDataset = ObjDataset(
    #                             root=train_path,
    #                             transform=transformAug,
    #                             img_size=(imageWidth, imageHeight),
    #                             normalize_boxes=True, useHFlip=True, useVFlip=True)

    train_datasets = [
        ObjDatasetV2(root=p, transform=transformAug, img_size=(imageWidth, imageHeight),
                normalize_boxes=True, useHFlip=True, useVFlip=True)
        for p in train_path
    ]
    trainDataset= ConcatDataset(train_datasets)
    
    # validDataset = ObjDataset(
    #                         root=val_path,
    #                         transform=transform,
    #                         img_size=(imageWidth, imageHeight),
    #                         normalize_boxes=True)
     
    
    val_datasets = [
        ObjDatasetV2(root=q, transform=transform, img_size=(imageWidth, imageHeight),
                normalize_boxes=True)
        for q in val_path
    ]
    validDataset= ConcatDataset(val_datasets)
       
    trainLoader = DataLoader(trainDataset, batch_size=trainBatchSize, shuffle=True, drop_last=True, collate_fn=lambda x: tuple(zip(*x)))
    validLoader = DataLoader(validDataset, batch_size=1, shuffle=True, drop_last=False, collate_fn=lambda x: tuple(zip(*x)))
    ID_TO_LABEL = {
        "OneDot": 0,
        "TwoDot": 1,
        "Ignore": 2,
    }
    
    LABEL_NAMES = {v: k for k, v in ID_TO_LABEL.items()}
    
    # print(f"Number of training samples: {len(trainDataset)}")
    # print(f"Number of validation samples: {len(validDataset)}")

    logging.info(f"Number of training samples: {len(trainDataset)}")
    logging.info(f"Number of validation samples: {len(validDataset)}")
    flush_stdout()

    # Initialize model
    num_classes = 3
    model = ObjLarge(num_classes=num_classes)
    



    model = model.to(device)

    # Loss and optimizer
    loss_fn = ComputeLoss(model, params={'cls': 0.5, 'box': 12.5, 'dfl': 2.5})

    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
    ema = EMA(model)
    amp_scaler = torch.cuda.amp.GradScaler() 
    # Track best IoU
    best_iou = 0.0
    best_model_path = None
    best_model_onnx_path = None
    best_model_dom_path = None

    # # Directory to save model
    # if getattr(sys, 'frozen', False):
    #     # If running from a bundled executable
    #     current_dir = os.path.dirname(sys.executable)
    # else:
    #     # If running from a script
    #     current_dir = os.path.dirname(os.path.abspath(__file__))

   
    # dataset_name = os.path.basename(os.path.normpath(train_path))
    Base_dir = os.path.join(result_model_dir_path)

    os.makedirs(Base_dir, exist_ok=True) 
    # build_date = str(date.today())

    # existing_runs = [d for d in os.listdir(Base_dir) if os.path.isdir(os.path.join(Base_dir, d))]
    # run_number = len(existing_runs) + 1
    # run_dir = os.path.join(Base_dir, f"run_{run_number}")
    # os.makedirs(run_dir)
    
    ################## Datasets for Training and Validation ##################


    train_csvs = []
    for i, dataset in enumerate(train_datasets):
        sub_csv_path = os.path.join(Base_dir, f"train_data_{i+1}.csv")
        dataset.generate_csv(sub_csv_path)
        train_csvs.append(pd.read_csv(sub_csv_path))

    # Combine and save one CSV
    combined_train_csv = pd.concat(train_csvs, ignore_index=True)
    combined_train_csv.to_csv(os.path.join(Base_dir, "train_data_combined.csv"), index=False)

    val_csvs = []
    for i, dataset in enumerate(val_datasets):
        sub_csv_path_val = os.path.join(Base_dir, f"val_data_{i+1}.csv")
        dataset.generate_csv(sub_csv_path_val)
        val_csvs.append(pd.read_csv(sub_csv_path_val))

    # Combine and save one CSV
    combined_val_csv = pd.concat(val_csvs, ignore_index=True)
    combined_val_csv.to_csv(os.path.join(Base_dir, "val_data_combined.csv"), index=False)

    ######################################################################
    # Check if a checkpoint exists
    checkpoint_dir = os.path.join(Base_dir, 'Checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)


    start_epoch = 0
    if resume_training_path:
        if os.path.exists(resume_training_path):
            start_epoch = load_checkpoint(resume_training_path, model, optimizer) + 1
            logging.info(f"Resuming training from checkpoint: {resume_training_path}, starting at epoch {start_epoch}")
            flush_stdout()
        else:
            raise FileNotFoundError(f"Checkpoint not found: {resume_training_path}")
    else:
        logging.info("No checkpoint provided. Starting training from scratch.")
        flush_stdout()

    saving_metrics = []
    for epoch in range(start_epoch, epochs):
        train_class_iou_totals = {}  # To track IoU for each class
        train_class_box_counts = {}  # To track the number of GT boxes per class
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
                _, _, height, width = images.shape

                with torch.no_grad():
                    model.eval()
                    outputs = model(images)
                    #mean_iou, tr_vis_gt_boxes, tr_vis_pred_boxes, tr_vis_pred_labels, tr_vis_pred_scores, train_per_class_totals, train_per_class_counts = pixel_iou_from_nms(outputs, targets, width=width, height=height, visualize=False)
                    mean_iou, train_per_class_totals, train_per_class_counts = pixel_iou_from_nms(outputs, targets, width=width, height=height, visualize=False)

                    train_ious += mean_iou
                    
                for cls_id in train_per_class_totals:
                    train_class_iou_totals[cls_id] = train_class_iou_totals.get(cls_id, 0.0) + train_per_class_totals[cls_id]
                    train_class_box_counts[cls_id] = train_class_box_counts.get(cls_id, 0) + train_per_class_counts[cls_id]
                # visualize_predictions(
                # image=images[0],  # Just the first image in the batch
                # gt_boxes=np.array(tr_vis_gt_boxes),
                # pred_boxes=np.array(tr_vis_pred_boxes),
                # gt_labels=None,
                # pred_labels=tr_vis_pred_labels,
                # pred_scores=tr_vis_pred_scores
                # )



                    
                    
                model.train()    
                pbar.update(1)
                pbar.set_postfix(loss=loss.item(), ious=mean_iou)
        train_loss /= len(trainLoader)

        train_mean_iou =train_ious/ len(trainLoader)

                    
        #print(f"Train Loss:{train_loss:.9f}, Train IoU: {train_mean_iou:.9f}")
        logging.info(f"[Epoch {epoch}] Train Loss: {train_loss:.9f}, Train IoU: {train_mean_iou:.9f}")
        flush_stdout()

        train_per_class_iou = {}
        # Calculate and display per-class IoU
        # print("\nTrain Per-Class IoU:")
        for cls, iou_total in train_class_iou_totals.items():
            class_mean_iou = iou_total / train_class_box_counts[cls]
            class_name = LABEL_NAMES.get(cls, f"Class_{cls}")
            train_per_class_iou[f"Train {class_name} IoU"] = round(class_mean_iou, 9)
            print(f"Train {class_name} IoU: {class_mean_iou:.4f}")
            logging.info(f"[Epoch {epoch}] Train {class_name} IoU: {class_mean_iou:.4f}")
            flush_stdout()

        print("\n")
        
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
            best_model_path = os.path.join(Base_dir, f"BestModelWeight.pth")
            torch.save(ema.ema.state_dict(), best_model_path)
            # print(f"Saved best model at epoch {epoch} with train IoU : {best_iou:.4f}")


            if best_model_onnx_path is not None and os.path.exists(best_model_onnx_path):
                os.remove(best_model_onnx_path)

            if best_model_dom_path is not None and os.path.exists(best_model_dom_path):
                os.remove(best_model_dom_path)
                
            # Export to ONNX
            onnx_export_path = os.path.join(Base_dir, f"BestModel.onnx")
            dom_export_path = os.path.join(Base_dir, f"BestModel.dom")
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
                "classNames": ["OneDot", 'TwoDot', 'Ignore']
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


        val_class_iou_totals = {}  # To track IoU for each class
        val_class_box_counts = {}  # To track the number of GT boxes per class


        total_iou = 0.0
        total_boxes = 0

        model.eval()
        with torch.no_grad():
            for i, (images_val, targets_val) in enumerate(validLoader):
                # Prepare images and targets
                images_val = torch.stack([img.to(device) for img in images_val], dim=0)
                targets_val = [{k: v.to(device).float() for k, v in t.items()} for t in targets_val]

                _, _, height, width = images_val.shape  # Get the actual image dimensions after resizing

                    
                # Get model predictions
                outputs = model(images_val)


                mean_iou, vis_gt_boxes, vis_pred_boxes, vis_pred_labels, vis_pred_scores,val_per_class_totals, val_per_class_counts = pixel_iou_from_nms(
                    outputs=outputs,
                    targets=targets_val,
                    width=width,
                    height=height,
                    visualize=True
                )

                total_iou += mean_iou
                
                for cls_id in val_per_class_totals:
                    val_class_iou_totals[cls_id] = val_class_iou_totals.get(cls_id, 0.0) + val_per_class_totals[cls_id]
                    val_class_box_counts[cls_id] = val_class_box_counts.get(cls_id, 0) + val_per_class_counts[cls_id]
    
                # Visualize the image after processing all classes
                visualize_predictions(
                    image=images_val[0],  # Image tensor
                    gt_boxes=np.array(vis_gt_boxes),  # Ground truth boxes
                    pred_boxes=np.array(vis_pred_boxes),  # Predicted boxes
                    gt_labels=None,  # Ground truth labels are optional
                    pred_labels=vis_pred_labels,  # Predicted labels
                    pred_scores=vis_pred_scores  # Confidence scores
                )


        overall_iou = total_iou / len(validLoader)
        #print(f"Validation IoU: {overall_iou}")

        logging.info(f"[Epoch {epoch}] Validation IoU: {overall_iou:.9f}")
        flush_stdout()

        per_class_iou = {}
        # Calculate and display per-class IoU
        # print("\nValidation Per-Class IoU:")
        for cls, iou_total in val_class_iou_totals.items():
            class_mean_iou = iou_total / val_class_box_counts[cls]
            class_name = LABEL_NAMES.get(cls, f"Class_{cls}")
            per_class_iou[f"Validation {class_name} IoU"] = round(class_mean_iou, 9)
            #print(f"Validation {class_name} IoU: {class_mean_iou:.4f}")
            logging.info(f"[Epoch {epoch}] Validation {class_name} IoU: {class_mean_iou:.4f}")
            flush_stdout()

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
        csv_path = os.path.join(Base_dir, f"training_metrics.csv")
        metrics_df.to_csv(csv_path, index=False)
        plot_metrics(metrics_df, Base_dir, epoch)
            
    # # Save final model
    # final_model_path = os.path.join(run_dir, f"{data_type}_LAST_H{imageHeight}W{imageWidth}_{build_date}_E{epoch}_iou_{train_mean_iou:.4f}.pth")
    # final_model_path_onnx = os.path.join(run_dir, f"{data_type}_LAST_H{imageHeight}W{imageWidth}_{build_date}_E{epoch}_iou_{train_mean_iou:.4f}.onnx")
    # final_model_path_dom = os.path.join(run_dir, f"{data_type}_LAST_H{imageHeight}W{imageWidth}_{build_date}_E{epoch}_iou_{train_mean_iou:.4f}.dom")
    # torch.save(ema.ema.state_dict(), final_model_path)
    
    
    # dummy_input = torch.randn(1, 3, imageHeight, imageWidth).to(device)

    # # Use the EMA model for export
    # ema.ema.eval()
    # buffer = io.BytesIO()
    # torch.onnx.export(ema.ema,
    #                 dummy_input,
    #                 buffer,
    #                 export_params=True,
    #                 opset_version=17,
    #                 do_constant_folding=True,
    #                 input_names=['input'],
    #                 output_names=['output'])
    # buffer.seek(0)
    # bytes_data = buffer.read()

    # # Save ONNX to disk (optional)
    # with open(final_model_path_onnx, "wb") as f:
    #     f.write(bytes_data)


    # base64Model = base64.b64encode(bytes_data).decode('ascii')

    # # Create custom DSM JSON structure
    # hvs_model_json = {
    #     "inputHeight": imageHeight,
    #     "inputWidth": imageWidth,
    #     "module": base64Model,
    #     "headType": "yolo8",
    #     "classNames": ["OneDot", 'TwoDot', 'Ignore']
    # }

    # # Save DSM format
    # jsonFormatString = json.dumps(hvs_model_json)
    # with open(final_model_path_dom, "w") as text_file:
    #     text_file.write(jsonFormatString)
                
    # print(f"Saved final model after epoch {epoch}")


def plot_metrics(metrics_df, Base_dir, epoch):
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
    plt.savefig(os.path.join(Base_dir, f'IoU.png'))
    plt.close()
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_df['Epoch'], metrics_df['TrainLoss'], label='TrainLoss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Loss (Up to Epoch {epoch})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(Base_dir, f'Loss.png'))
    plt.close()

if __name__ == '__main__':
    run_train()
