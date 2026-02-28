# 🚗 Autonomous Driving — Car Detection with YOLO (Deep Study Project)

---

## 📌 Project Overview

This project is a **deep, academic implementation and exploration of YOLO-based object detection**, built for:

- 📚 Study and conceptual mastery
- 🧠 Interview preparation
- 🔬 Research-level understanding
- 🛠 Modern TensorFlow 2 implementation practice

The goal is not just to "run YOLO", but to:

- Understand how YOLO works mathematically
- Rebuild the post-processing pipeline from scratch
- Compare academic implementation vs production-ready detectors

---

## 🧠 What This Project Covers

### 1️⃣ Manual YOLOv2 Post-Processing

We implement from scratch:

- Bounding box decoding
- Coordinate conversion
- Score filtering
- Intersection over Union (IoU)
- Non-Max Suppression (NMS)
- Box scaling to original image

All written in **TensorFlow 2 (Eager Execution)**:

- No sessions
- No TF1 graph mode
- No deprecated backend calls

---

### 2️⃣ Mathematical Foundations

#### Bounding Box Parameterization

\[
b_x = \sigma(t_x) + c_x, \quad b_y = \sigma(t_y) + c_y
\]

\[
b_w = p_w e^{t_w}, \quad b_h = p_h e^{t_h}
\]

Where:

- \( (c_x, c_y) \) = grid cell location
- \( (p_w, p_h) \) = anchor dimensions
- \( \sigma \) = sigmoid

---

#### Score Computation

\[
\text{Class Score}\_i = P_c \times P(\text{class}\_i \mid \text{object})
\]

Final score:

\[
\max_i \Big(P_c \cdot P(\text{class}\_i \mid \text{object})\Big)
\]

Thresholding removes background and low-confidence detections.

---

#### Intersection over Union (IoU)

\[
IoU = \frac{\text{Area of Intersection}}{\text{Area of Union}}, \quad \text{Union} = A + B - \text{Intersection}
\]

- IoU = 0 → no overlap
- IoU = 1 → perfect overlap

---

#### Non-Max Suppression (NMS)

Algorithm:

1. Sort boxes by score
2. Select highest
3. Remove overlapping boxes (IoU > threshold)
4. Repeat

Implemented using:

- `tf.image.non_max_suppression`

---

## 🏗 Project Structure

```text
Autonomous-Driving-Car-Detection/
│
├── notebooks/
│   ├── YOLO_v2_TF2_Implementation.ipynb
│   ├── Car_Detection_YOLOV8_Implementation.ipynb
│
├── yolo_utils.py
├── README.md
└── model_data/
```

---

## 🔬 Implementation Pipeline (YOLOv2 Style)

```text
CNN Output (19×19×5×85)
↓
Convert (xywh → x1,y1,x2,y2)
↓
Score Filtering
↓
Scale to Original Image
↓
Non-Max Suppression
↓
Final Predictions
```

---

## ✅ Final Summary

- YOLO predicts offsets relative to grid cells and anchors.
- Objectness × class probability gives final class score.
- Thresholding reduces computational burden.
- IoU quantifies spatial overlap.
- NMS removes duplicate detections.
- Scaling ensures boxes align with original image dimensions.

Modern YOLO versions (e.g., YOLOv8) handle these steps internally, but **understanding the full pipeline is essential for research, debugging, and interviews**.

---

---

# 🚀 KerasCV YOLOv8

This project integrates **KerasCV’s YOLOv8 implementation** to demonstrate how modern object detection systems abstract complex post-processing steps into a clean, production-ready API.

KerasCV’s YOLOv8 includes:

- ✅ Built-in decoding of raw predictions
- ✅ Built-in Non-Max Suppression (NMS)
- ✅ Pretrained weights (Pascal VOC / COCO presets)
- ✅ Fully TensorFlow 2 compatible pipeline
- ✅ Production-ready inference workflow

This demonstrates the evolution from:

> Manual academic YOLOv2 implementation → Modern industry-ready YOLOv8 deployment.

---

# 📦 Requirements

## Environment

- Python 3.10+
- TensorFlow 2.x
- NumPy
- Matplotlib
- Pillow
- keras-cv

---

## Installation

```bash
pip install tensorflow keras-cv matplotlib pillow numpy
```

## 🎯 Learning Outcomes

After completing this project, you will understand:

### Detection Theory

- ✔ Anchor box mechanics
- ✔ Grid responsibility assignment in YOLO
- ✔ Why sigmoid is applied to center predictions
- ✔ How objectness and class probabilities combine

### Mathematical Foundations

- ✔ How Intersection over Union (IoU) is computed
- ✔ How Non-Max Suppression (NMS) works internally
- ✔ Why score thresholding is necessary

### Practical Engineering

- ✔ Differences between research implementations and production APIs
- ✔ How modern libraries abstract decoding and NMS
- ✔ How to deploy pretrained object detectors efficiently

This goes far beyond simply “running a model.”

---

## 📈 Why This Project Matters

Many practitioners:

- Use pretrained detectors
- Never understand bounding box decoding
- Never compute IoU manually
- Never implement NMS from scratch

This project ensures deep understanding of **Object Detection Internals**, which is critical for:

- Research work
- Model customization
- Architecture design
- Performance optimization
- Technical interviews

**Understanding internals gives you control.  
Using modern APIs gives you speed.  
You need both.**

---

## 🔮 Future Improvements

Possible extensions for advanced exploration:

- Class-wise NMS implementation
- Soft-NMS
- CIoU / DIoU loss analysis
- Custom dataset fine-tuning
- ONNX export
- TensorRT deployment
- Real-time inference benchmarking
- Model quantization
- Edge deployment experiments

---

## 🧾 Final Note

This repository follows a:

> **Study-first, production-second philosophy.**

It is structured for:

- Long-term reference
- Deep conceptual clarity
- Strong theoretical foundation
- Practical modern deployment skills

It bridges:

**Academic Understanding → Real-World Implementation**

---

## 👨‍💻 Author

Deep learning exploration project focused on mastering object detection systems from fundamentals to production-level deployment.

---

## 📚 References

- YOLOv2 Paper
- YOLOv8 Documentation
- TensorFlow 2 API Documentation
- COCO Dataset Documentation
- Pascal VOC Dataset Documentation
