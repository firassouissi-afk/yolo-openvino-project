# 🚀 YOLO OpenVINO Real-Time Object Detection

![Python](https://img.shields.io/badge/python-3.11-blue)
![OpenVINO](https://img.shields.io/badge/OpenVINO-Accelerated-orange)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-green)
![Platform](https://img.shields.io/badge/Windows-Compatible-lightgrey)
![License](https://img.shields.io/badge/license-MIT-yellow)

---

## 📌 Overview

This project is a **real-time object detection system** using **YOLOv8n optimized with Intel OpenVINO**.

It is designed to run efficiently on **low-power hardware (Intel UHD Graphics / CPU)** while maintaining smooth real-time performance.

---

## 🎯 Features

- ⚡ OpenVINO accelerated inference (CPU / Intel GPU auto)
- 🎥 Real-time webcam object detection
- 📊 FPS counter + CPU usage overlay
- 🧵 Multi-threaded inference pipeline
- 🔥 Lightweight YOLOv8n model (fast & efficient)
- 🖥 Optimized for Intel UHD Graphics (12th Gen tested)

---

## 🧠 How It Works

1. Webcam captures frames using OpenCV
2. Frames are sent to YOLOv8 OpenVINO model
3. OpenVINO runs inference (CPU or Intel GPU automatically)
4. Bounding boxes + labels are drawn on detected objects
5. FPS + CPU usage are displayed in real time

---

## ⚙️ Installation
pip install ultralytics opencv-python psutil
pip install openvino
yolo export model=yolov8n.pt format=openvino

### 1. Create virtual environment (recommended)

```bash
python -m venv ai
ai\Scripts\activate
