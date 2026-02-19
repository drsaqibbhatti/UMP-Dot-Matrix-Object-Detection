import os
import json
import random
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import csv
class ObjDataset(Dataset):
    def __init__(self, root, transform=None, useHFlip=False, useVFlip=False, img_size=(512, 512), normalize_boxes=True):

        self.root = root  # Directly use the path passed

        self.transform = transform
        self.useHFlip = useHFlip
        self.useVFlip = useVFlip
        self.img_size = img_size
        self.normalize_boxes = normalize_boxes

        self.image_files = []
        for dirpath, _, filenames in os.walk(self.root):
            for f in filenames:
                if f.lower().endswith(('.jpg', '.png')):
                    base_name = os.path.splitext(f)[0]
                    json_path = os.path.join(dirpath, base_name + '.json')
                    img_path = os.path.join(dirpath, f)

                    if not os.path.exists(json_path):
                        continue

                    with open(json_path, 'r', encoding='utf-8') as jf:
                        data = json.load(jf)

                    if data.get('LabelCollection'):
                        self.image_files.append(img_path)

        print(f"[INFO] Collected {len(self.image_files)} images from {self.root}")



    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        base_name = os.path.splitext(img_name)[0]
        json_name = base_name + '.json'

        img_path = os.path.join(self.root, img_name)
        json_path = os.path.join(self.root, json_name)

        # Load image
        img = Image.open(img_path).convert('RGB')
        orig_width, orig_height = img.size

        # Load JSON
        with open(json_path, 'r',encoding="utf-8") as f:
            data = json.load(f)
        targets = data.get('LabelCollection', [])


        # Resize image
        img = img.resize(self.img_size, Image.NEAREST)



        # Compute scaling
        scale_x = self.img_size[1] / orig_width
        scale_y = self.img_size[0] / orig_height

        boxes = []
        labels = []


        LABEL_MAP = {
            "onedot": 0,
            "twodot": 1
        }
        
        for target in targets:
            
            label_name = target["LabelName"]
            class_id = LABEL_MAP.get(label_name)

            if class_id is None:
                continue  # Skip unknown labels
            
            x_min = target['X']
            y_min = target['Y']
            width = target['Width']
            height = target['Height']
            x_max = x_min + width
            y_max = y_min + height

            # Scale bounding box coordinates to resized image dimensions
            x_min = x_min * scale_x
            y_min = y_min * scale_y
            x_max = x_max * scale_x
            y_max = y_max * scale_y

            # Convert to center format
            x_center = (x_min + x_max) / 2
            y_center = (y_min + y_max) / 2
            w = x_max - x_min
            h = y_max - y_min

            if self.normalize_boxes:
                x_center /= self.img_size[1]  # Normalize by target width
                y_center /= self.img_size[0]  # Normalize by target height
                w /= self.img_size[1]         # Normalize width
                h /= self.img_size[0]         # Normalize height

            boxes.append([x_center, y_center, w, h])
            labels.append(class_id)  # all "OneDot" → class 0

        # Apply optional flips
        if self.useHFlip and random.random() > 0.5:
            img = ImageOps.mirror(img)
            for i in range(len(boxes)):
                boxes[i][0] = 1.0 - boxes[i][0]
                        
        if self.useVFlip and random.random() > 0.5:
            img = ImageOps.flip(img)
            for i in range(len(boxes)):
                boxes[i][1] = 1.0 - boxes[i][1]  # Flip center Y

            

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        if self.transform:
            img = self.transform(img)
            
        # Convert image to BGR
        img = img[[2, 1, 0], :, :]  # RGB to BGR

        target = {'boxes': boxes, 'labels': labels}
        return img, target


    def generate_csv(self, csv_filename):
        """
        Generates a CSV file with image name and associated category IDs.
        Format: image_name, category_ids (comma-separated)
        """
        
        LABEL_MAP = {
            "onedot": 0,
            "twodot": 1
        }
        
        
        with open(csv_filename, mode='w', newline='', encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(['image_name', 'category_ids'])

            for img_name in self.image_files:
                base_name = os.path.splitext(img_name)[0]
                json_name = base_name + '.json'
                json_path = os.path.join(self.root, json_name)

                if not os.path.exists(json_path):
                    continue

                with open(json_path, 'r', encoding="utf-8") as f:
                    data = json.load(f)
                targets = data.get('LabelCollection', [])

                # Assuming single class: OneDot → class 0
                category_ids = set()
                for ann in targets:
                    label_name = ann["LabelName"]
                    class_id = LABEL_MAP.get(label_name)
                    if class_id is not None:
                        category_ids.add(class_id)

                writer.writerow([img_name, ",".join(map(str, sorted(category_ids)))])

        print(f"CSV file saved to {csv_filename}")
