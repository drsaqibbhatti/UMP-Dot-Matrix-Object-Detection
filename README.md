# UMP Dot Matrix Pattern Detection (0/1) — Small Object Detection (YOLOv8-based)

This project detects **micro dot-pattern units** and a **long bar pattern** on an industrial surface and classifies each detected unit
Note: The original pattern specification and dataset are confidential.  

> This repository focuses on the **engineering approach** and **model pipeline** for small-object detection.

---

Project page (demo + overview):

https://drsaqibbhatti.com/projects/ump-dot-matrix-pattern.html

## Project Overview

### Goal
- Detect tiny pattern units (dot clusters / bar-like marks)
- Predict label per unit and generate the pattern of circles and bar
- Provide bounding box + confidence overlays for verification

### Why it’s hard
- Objects are **extremely small**
- Dense layouts + low contrast surfaces
- Small targets easily disappear when downscaled

---

## Key Features
- YOLOv8-style object detection pipeline customized for **small objects**
- Small-object oriented improvements:
  - Higher input resolution / multi-scale training
  - Detection head / anchor tuning for small targets
  - Enhanced feature fusion for shallow layers
  - Augmentations for robustness (brightness, blur, noise, rotation)
  - Optional tiling/ROI inference for maximum detail retention

---

## Tech Stack
- **Python**
- **PyTorch**
- **OpenCV**, NumPy
- YOLOv8-style architecture (custom modifications)

