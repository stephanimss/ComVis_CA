# realtime_detector.py
# Object Detection Real-Time menggunakan YOLOv8 dan OpenCV

import cv2
from ultralytics import YOLO
import time

# 1. Load YOLO Model
try:
    print("Loading YOLOv8n model...")
    model = YOLO("yolov8n.pt")  # model ringan untuk real-time
except Exception as e:
    print(f"Failed to load YOLO model: {e}")
    exit()

# 2. Inisialisasi Kamera
cap = cv2.VideoCapture(1)  # 0 = webcam default

if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

print("Camera accessed successfully.")
print("Detection started. Press 'Q' or 'ESC' to stop.")

prev_time = 0  # untuk menghitung FPS

# 3. Loop Deteksi Real-Time
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    # Hitung FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time else 0
    prev_time = current_time

    # YOLO inference
    results = model(frame, stream=True, verbose=False)

    annotated_frame = frame
    for r in results:
        annotated_frame = r.plot()

    # Tampilkan FPS pada frame
    cv2.putText(
        annotated_frame,
        f"FPS: {fps:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # Tampilkan hasil deteksi
    cv2.imshow("YOLOv8 Real-Time Detection", annotated_frame)

    key = cv2.waitKey(1)
    if key == ord('q') or key == 27:  # ESC
        print("Detection stopped by user.")
        break

# 4. Bersihkan Resource
cap.release()
cv2.destroyAllWindows()
print("Resources released. Program finished.")
