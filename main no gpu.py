from ultralytics import YOLO
import cv2
import time

# Load model
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(0)

# FPS calculation variables
prev_time = 0
curr_time = 0

print("Starting YOLO with FPS overlay... Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # =========================
    # FPS CALCULATION
    # =========================
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    # =========================
    # YOLO DETECTION
    # =========================
    results = model(frame)
    annotated_frame = results[0].plot()

    # =========================
    # OVERLAY PANEL
    # =========================
    height, width, _ = frame.shape

    # Background box for stats
    cv2.rectangle(
        annotated_frame,
        (10, 10),
        (280, 110),
        (0, 0, 0),
        -1
    )

    # FPS text
    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.2f}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    # Resolution
    cv2.putText(
        annotated_frame,
        f"Resolution: {width}x{height}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1
    )

    # Model info
    cv2.putText(
        annotated_frame,
        "Model: YOLOv8n (nano)",
        (20, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1
    )

    # =========================
    # SHOW FRAME
    # =========================
    cv2.imshow("YOLO AI System", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()