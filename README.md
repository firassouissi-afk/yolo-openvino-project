YOLOv8 OpenVINO Real-Time Object Detection (Intel GPU/CPU Optimized)

============================================================

OVERVIEW
------------------------------------------------------------
This project is a real-time object detection system using:

- YOLOv8 (Ultralytics)
- OpenVINO backend (Intel acceleration)
- OpenCV video streaming
- Multi-threaded inference pipeline
- FPS + CPU performance overlay

It is optimized for Intel UHD Graphics (12th Gen) with CPU fallback.

============================================================

FEATURES
------------------------------------------------------------
- Real-time object detection (YOLOv8n)
- OpenVINO acceleration (CPU / Intel GPU auto)
- FPS counter overlay
- CPU usage monitoring
- Multi-threaded inference (smooth video)
- Webcam input (640x480 optimized)
- Lightweight and fast

============================================================

HOW IT WORKS
------------------------------------------------------------
1. Webcam captures frames using OpenCV
2. Frames are sent to a background inference thread
3. YOLOv8 OpenVINO model processes frames
4. Detection results are drawn using .plot()
5. Main thread displays video with performance overlay

============================================================

INSTALLATION
------------------------------------------------------------

1. Install Python
Make sure Python 3.11+ is installed

Check:
python --version


------------------------------------------------------------

2. Create virtual environment (recommended)

python -m venv ai
ai\Scripts\activate


------------------------------------------------------------

3. Install dependencies

pip install ultralytics opencv-python psutil


------------------------------------------------------------

4. Install OpenVINO runtime

pip install openvino


============================================================

EXPORT MODEL (OPTIONAL)
------------------------------------------------------------
If you want to export YOLOv8 to OpenVINO format:

yolo export model=yolov8n.pt format=openvino

This creates:
yolov8n_openvino_model/


============================================================

HOW TO RUN
------------------------------------------------------------
Run the script:

python main.py


Press:
Q → Quit program


============================================================

PERFORMANCE TIPS
------------------------------------------------------------
For best FPS on Intel UHD:

- Use yolov8n (nano model only)
- Keep resolution at 640x480 or lower
- Enable threading (already included)
- Do NOT use CUDA (Intel UHD does not support it in PyTorch)
- OpenVINO automatically uses best available backend

============================================================

SUPPORTED HARDWARE
------------------------------------------------------------
- Intel UHD Graphics (12th Gen)
- Intel Iris Xe
- CPU fallback mode
- Any OpenVINO-compatible device

============================================================

NOTES
------------------------------------------------------------
- CUDA is not used in this project
- OpenVINO automatically selects CPU/GPU
- Threading improves smoothness
- Best performance depends on Intel optimization

============================================================

AUTHOR
------------------------------------------------------------
Firas Souissi
Computer Engineering Student
Simulation Pilot (Flight Sim / DCS / MSFS)
