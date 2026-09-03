import cv2
import numpy as np
from ultralytics import YOLO
import requests
import time
import base64
from datetime import datetime
import urllib3
from collections import defaultdict

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ═══════════════════════════════════════════════════════════
# ⚙️ تنظیمات و آدرس ngrok
# ═══════════════════════════════════════════════════════════
BASE_NGROK_URL   = "https://outfit-dimly-juice.ngrok-free.dev"
API_URL          = f"{BASE_NGROK_URL}/api/violations/camera"

CAMERA_NAME      = "دوربین ۲ - خط تولید A"
VIDEO_SOURCE     = "vid2.mp4"
COOLDOWN_SECONDS = 30
CONF_THRESHOLD   = 0.45   # حداقل درصد اطمینان برای تشخیص انسان
CONFIRM_FRAMES   = 3      # تعداد فریم متوالی جهت تأیید واقعی بودن شخص

# ═══════════════════════════════════════════════════════════
# 🧠 بارگذاری مدل و پیکربندی
# ═══════════════════════════════════════════════════════════
print("=" * 70)
print("🧠 موتور پردازش تصویر هوشمند (دوربین ۲) فعال شد...")
print("=" * 70)

# مدل yolov8s دقت بسیار بهتری نسبت به yolov8n دارد
model = YOLO("yolov8s.pt")

cap = cv2.VideoCapture(VIDEO_SOURCE)

last_alert_time    = 0
is_person_in_frame = False
confirm_counter    = 0

def convert_frame_to_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{jpg_as_text}"

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        is_person_in_frame = False
        confirm_counter = 0
        continue

    # اجرای تشخیص با YOLO
    results = model(frame, conf=CONF_THRESHOLD, verbose=False)
    current_frame_has_person = False
    alert_name = ""

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            conf = float(box.conf[0])
            
            if class_name == "person" and conf >= CONF_THRESHOLD:
                current_frame_has_person = True
                alert_name = "عدم استفاده از تجهیزات ایمنی / ورود به منطقه خط تولید"
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # رسم کادر تشخیص شخص روی فریم
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"Person {conf:.0%}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # سیستم تأیید چند فریمی برای حذف نویز
    if current_frame_has_person:
        confirm_counter = min(confirm_counter + 1, CONFIRM_FRAMES + 1)
    else:
        confirm_counter = max(confirm_counter - 1, 0)

    person_confirmed = (confirm_counter >= CONFIRM_FRAMES)
    current_time = time.time()

    # ارسال هشدار به API
    if person_confirmed:
        if not is_person_in_frame and (current_time - last_alert_time > COOLDOWN_SECONDS):
            try:
                image_base64 = convert_frame_to_base64(frame)
                payload = {
                    "violationType": alert_name,
                    "imageUrl": image_base64,
                    "cameraLocation": CAMERA_NAME,
                    "detectedAt": datetime.now().isoformat(),
                    "hseComment": f"شناسایی خودکار توسط هوش مصنوعی Colab روی {CAMERA_NAME}"
                }
                
                response = requests.post(API_URL, json=payload, verify=False, timeout=5)
                
                if response.status_code in [200, 201]:
                    print(f"🎯 [{datetime.now():%H:%M:%S}] تخلف ثبت شد: {alert_name}")
                    last_alert_time = current_time
                    is_person_in_frame = True
                else:
                    print(f"⚠️ پاسخ دات‌نت: {response.status_code}")
                    
            except Exception as e:
                print(f"❌ خطا در ارسال داده به API: {e}")
    else:
        if current_time - last_alert_time > COOLDOWN_SECONDS:
            is_person_in_frame = False

    time.sleep(0.03)

cap.release()
print("✅ پردازش تمام شد")