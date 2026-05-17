import cv2
import time
import psutil
import threading
from ultralytics import YOLO

# =============================
# MODEL (OpenVINO ONLY)
# =============================
model = YOLO(r"C:\Users\FIRAS\Desktop\py rec\yolov8n_openvino_model", task="detect")
model.overrides["verbose"] = False

print("Using OpenVINO backend (CPU/GPU auto)")

# =============================
# CAMERA
# =============================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# =============================
# GLOBALS
# =============================
frame = None
output = None
lock = threading.Lock()

fps_avg = 0
prev_time = time.time()

# =============================
# INFERENCE THREAD (NO CUDA, NO DEVICE OVERRIDE)
# =============================
def run_inference():
    global frame, output

    while True:
        if frame is None:
            continue

        with lock:
            img = frame.copy()

        # OpenVINO inference (CPU or GPU automatically handled)
        results = model(img, verbose=False)

        with lock:
            output = results[0].plot()

# start thread
threading.Thread(target=run_inference, daemon=True).start()

# =============================
# MAIN LOOP
# =============================
while True:
    ret, new_frame = cap.read()
    if not ret:
        break

    with lock:
        frame = new_frame

        display = output if output is not None else new_frame

    # =============================
    # FPS
    # =============================
    curr = time.time()
    fps = 1 / (curr - prev_time)
    prev_time = curr

    fps_avg = fps if fps_avg == 0 else (0.9 * fps_avg + 0.1 * fps)

    cpu = psutil.cpu_percent()

    # =============================
    # OVERLAY
    # =============================
    cv2.putText(display, f"FPS: {fps_avg:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(display, f"CPU: {cpu}%", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.putText(display, "OpenVINO (CPU/GPU auto)", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    # =============================
    # SHOW
    # =============================
    cv2.imshow("YOLO OpenVINO FIXED", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()