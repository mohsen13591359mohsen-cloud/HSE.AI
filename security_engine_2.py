import cv2
import requests
import base64
import time
import urllib3
import threading
import torch
from flask import Flask, Response
from flask_cors import CORS
from ultralytics import YOLO

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
torch.set_num_threads(2)

# 🎯 نکته مهم: آدرس ngrok سیستم خود را جایگزین کنید!

BASE_NGROK_URL = "https://a1b2-c3d4-e5f6.ngrok-free.app"  # آدرس اختصاصی شما
API_URL = f"{BASE_NGROK_URL}/api/security/incidents"

VIDEO_PATH = "vid2.mp4"
CAMERA_ID = "CAM-NORTH-03"
LOCATION = "انبار مرکزی - دوربین ۳"
COOLDOWN_SECONDS = 3
last_alarm_time = 0

app = Flask(__name__)
CORS(app)

global_jpeg_bytes = None

print("⏳ در حال بارگذاری مدل YOLOv8...")
model = YOLO("yolov8n.pt")

def frame_to_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

def send_alarm_to_api(frame, alarm_type):
    global last_alarm_time
    if time.time() - last_alarm_time < COOLDOWN_SECONDS:
        return
    last_alarm_time = time.time()
    
    payload = {
        "cameraId": CAMERA_ID,
        "location": LOCATION,
        "alarmType": alarm_type,
        "imageBase64": frame_to_base64(frame)
    }

    def _send():
        try:
            requests.post(API_URL, json=payload, verify=False, timeout=3)
            print(f"🚨 [هشدار دزدگیر] ثبت شد: {alarm_type}")
        except Exception as e:
            print(f"💥 خطای ارتباط با API دات‌نت: {e}")

    threading.Thread(target=_send, daemon=True).start()

def process_video_main():
    global global_jpeg_bytes
    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"❌ خطای بحرانی: فایل ویدیویی '{VIDEO_PATH}' پیدا نشد!")
        return

    print(f"🎬 پردازش ویدیو '{VIDEO_PATH}' در پس‌زمینه کولاب شروع شد...")

    frame_count = 0
    last_boxes = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_count += 1

        if frame_count % 2 == 0:
            results = model(frame, verbose=False)
            last_boxes = []
            person_detected = False

            for r in results:
                for box in r.boxes:
                    if model.names[int(box.cls[0])] == "person" and float(box.conf[0]) > 0.50:
                        person_detected = True
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])
                        last_boxes.append((x1, y1, x2, y2, conf))

            if person_detected:
                send_alarm_to_api(frame, "ورود غیرمجاز (تشخیص هوش مصنوعی)")

        for (x1, y1, x2, y2, conf) in last_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"INTRUDER {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        success, encoded_img = cv2.imencode('.jpg', frame)
        if success:
            global_jpeg_bytes = encoded_img.tobytes()

        time.sleep(0.03)

    cap.release()

def generate_mjpeg():
    global global_jpeg_bytes
    while True:
        if global_jpeg_bytes is None:
            time.sleep(0.05)
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + global_jpeg_bytes + b'\r\n')
        time.sleep(0.04)

@app.route('/video_feed')
def video_feed():
    return Response(generate_mjpeg(), mimetype='multipart/x-mixed-replace; boundary=frame')

def run_flask_app():
    app.run(host="0.0.0.0", port=5005, debug=False, use_reloader=False, threaded=True)

if __name__ == "__main__":
    flask_thread = threading.Thread(target=run_flask_app, daemon=True)
    flask_thread.start()
    print("📡 استریم زنده روی پورت 5005 فعال شد.")
    process_video_main()