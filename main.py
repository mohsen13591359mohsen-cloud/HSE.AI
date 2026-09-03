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

# ═══════════════════════════════════════════════════════════
# ⚙️ تنظیمات و آدرس ngrok
# ═══════════════════════════════════════════════════════════
BASE_NGROK_URL   = "https://outfit-dimly-juice.ngrok-free.dev"
API_URL          = f"{BASE_NGROK_URL}/api/violations/camera"

CAMERA_NAME      = "دوربین ۲ - خط تولید A"
VIDEO_SOURCE     = "vid2.mp4"
COOLDOWN_SECONDS = 30
CONF_THRESHOLD   = 0.45   # حداقل درصد اطمینان برای تشخیص
CONFIRM_FRAMES   = 3      # تعداد فریم متوالی جهت تأیید واقعی تخلف

# ═══════════════════════════════════════════════════════════
# 🧠 بارگذاری مدل اختصاصی PPE (Protective Equipment)
# ═══════════════════════════════════════════════════════════
print("=" * 70)
print("👝 موتور هوشمند تشخیص عدم استفاده از تجهیزات ایمنی (PPE) فعال شد...")
print("=" * 70)

# دانلود خودکار مدل اختصاصی PPE در صورت عدم وجود در مسیر
MODEL_PATH = "ppe_yolov8.pt"
if not os.path.exists(MODEL_PATH):
    print("⏳ در حال دریافت مدل اختصاصی تشخیص PPE...")
    import urllib.request
    url = "https://huggingface.co/keremberke/yolov8s-protective-equipment-detection/resolve/main/model.pt"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("✅ مدل PPE با موفقیت دانلود و آماده شد.")

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_SOURCE)

last_alert_time     = 0
is_violation_active = False
confirm_counter     = 0

def convert_frame_to_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{jpg_as_text}"

# کلیدواژه‌های مربوط به عدم استفاده از تجهیزات در مدل‌های PPE
VIOLATION_KEYWORDS = ["no-hardhat", "no-helmet", "no-safety vest", "no-vest", "no-mask", "no-gloves"]

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        is_violation_active = False
        confirm_counter = 0
        continue

    # اجرای مدل روی فریم
    results = model(frame, conf=CONF_THRESHOLD, verbose=False)
    current_frame_has_violation = False
    detected_violations = []

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            conf = float(box.conf[0])
            
            # بررسی اینکه آیا کلاس شناسایی‌شده جزء تخلفات عدم استفاده از PPE است یا خیر
            if any(vk in class_name.lower() for vk in VIOLATION_KEYWORDS):
                current_frame_has_violation = True
                detected_violations.append(class_name)
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # رسم کادر قرمز اختصاصی دور تخلف PPE
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"Violation: {class_name} ({conf:.0%})", 
                            (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # فیلتر تأیید چند فریمی جهت حذف نویز و False Positive
    if current_frame_has_violation:
        confirm_counter = min(confirm_counter + 1, CONFIRM_FRAMES + 1)
    else:
        confirm_counter = max(confirm_counter - 1, 0)

    violation_confirmed = (confirm_counter >= CONFIRM_FRAMES)
    current_time = time.time()

    # ارسال هشدار به API پروژه دات‌نت
    if violation_confirmed:
        if not is_violation_active and (current_time - last_alert_time > COOLDOWN_SECONDS):
            alert_name = f"عدم استفاده از تجهیزات ایمنی ({', '.join(set(detected_violations))})"
            try:
                image_base64 = convert_frame_to_base64(frame)
                payload = {
                    "violationType": alert_name,
                    "imageUrl": image_base64,
                    "cameraLocation": CAMERA_NAME,
                    "detectedAt": datetime.now().isoformat(),
                    "hseComment": f"شناسایی خودکار عدم رعایت PPE توسط AI روی {CAMERA_NAME}"
                }
                
                response = requests.post(API_URL, json=payload, verify=False, timeout=5)
                
                if response.status_code in [200, 201]:
                    print(f"🎯 [{datetime.now():%H:%M:%S}] تخلف PPE ثبت شد: {alert_name}")
                    last_alert_time = current_time
                    is_violation_active = True
                elif response.status_code == 502:
                    print("⚠️ خطای 502 Bad Gateway: پروژه دات‌نت لوکال شما متصل نیست یا ngrok قطع شده است.")
                else:
                    print(f"⚠️ پاسخ دات‌نت: {response.status_code}")
                    
            except Exception as e:
                print(f"❌ خطا در ارسال داده به API: {e}")
    else:
        if current_time - last_alert_time > COOLDOWN_SECONDS:
            is_violation_active = False

    time.sleep(0.03)

cap.release()
print("✅ پردازش تمام شد")