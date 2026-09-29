import cv2
import numpy as np
from ultralytics import YOLO
import requests
import time
import base64
from datetime import datetime
import urllib3
import os

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ⚙️ تنظیمات
BASE_NGROK_URL   = "https://outfit-dimly-juice.ngrok-free.dev"
API_URL          = f"{BASE_NGROK_URL}/api/Violations/camera"

CAMERA_NAME      = "دوربین ۲ - خط تولید A"
VIDEO_SOURCE     = "lifterac.mp4"
COOLDOWN_SECONDS = 30
CONF_THRESHOLD   = 0.35   # آستانه اطمینان تشخیص برای best.pt
CONFIRM_FRAMES   = 3

print("=" * 70)
print("🚀 بارگذاری مدل اختصاصی HSE (best.pt)...")
print("=" * 70)

if not os.path.exists("best.pt"):
    print("❌ خطای حیاتی: فایل best.pt در پوشه برنامه پیدا نشد!")
    exit(1)

# 🧠 بارگذاری وزن‌های اختصاصی آموزش‌دیده
model = YOLO("best.pt")
cap = cv2.VideoCapture(VIDEO_SOURCE)

last_alert_time     = 0
is_violation_active = False
confirm_counter     = 0

def convert_frame_to_base64(frame, max_width=640):
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / float(w)
        frame = cv2.resize(frame, (max_width, int(h * scale)))
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
    return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        is_violation_active = False
        confirm_counter = 0
        continue

    # اجرای استنتاج مستقیم با مدل اختصاصی
    results = model(frame, conf=CONF_THRESHOLD, verbose=False)
    current_frame_has_violation = False

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # تشخیص مستقیم عدم استفاده از کلاه ایمنی از طریق مدل
            if class_name in ["No Helmet", "no_helmet"]:
                current_frame_has_violation = True
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"No Helmet ({conf:.0%})", 
                            (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                # سایر کلاس‌ها (Person, Forklift و ...)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(frame, f"{class_name} ({conf:.0%})", 
                            (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # تأیید چند فریمی برای جلوگیری از آلارم کاذب
    if current_frame_has_violation:
        confirm_counter = min(confirm_counter + 1, CONFIRM_FRAMES + 1)
    else:
        confirm_counter = max(confirm_counter - 1, 0)

    violation_confirmed = (confirm_counter >= CONFIRM_FRAMES)
    current_time = time.time()

    if violation_confirmed:
        if not is_violation_active and (current_time - last_alert_time > COOLDOWN_SECONDS):
            alert_name = "عدم استفاده از کلاه ایمنی در خط تولید"
            try:
                payload = {
                    "violationType": alert_name,
                    "imageUrl": convert_frame_to_base64(frame),
                    "cameraLocation": CAMERA_NAME,
                    "detectedAt": datetime.now().isoformat(),
                    "hseComment": f"شناسایی هوشمند عدم استفاده از کلاه ایمنی روی {CAMERA_NAME}"
                }
                response = requests.post(API_URL, json=payload, verify=False, timeout=5)
                if response.status_code in [200, 201]:
                    print(f"🎯 [{datetime.now():%H:%M:%S}] تخلف ثبت شد!")
                    last_alert_time = current_time
                    is_violation_active = True
            except Exception as e:
                print(f"❌ خطا در ارسال: {e}")
                last_alert_time = current_time
    else:
        if current_time - last_alert_time > COOLDOWN_SECONDS:
            is_violation_active = False

    # نمایش فریم (اختیاری)
    # cv2.imshow("HSE Real-time Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
# cv2.destroyAllWindows()