import cv2
import numpy as np
from ultralytics import YOLO
import requests
import time
import base64
import json
import os
from datetime import datetime
import urllib3
from IPython.display import display, Image, clear_output

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ═══════════════════════════════════════════════════════════
# ⚙️ تنظیمات اصلی
# ═══════════════════════════════════════════════════════════
BASE_NGROK_URL   = "https://outfit-dimly-juice.ngrok-free.dev"
API_URL          = f"{BASE_NGROK_URL}/api/SafetyIncidents/camera"

CAMERA_NAME      = "دوربین حریم اختصاصی - street"
VIDEO_SOURCE     = "/content/HSE.AI/street.mp4"
CONFIG_FILE      = "/content/HSE.AI/zone_config.json"
COOLDOWN_SECONDS = 15
CONF_THRESHOLD   = 0.40

CLASSES_PERSON  = ["person"]
CLASSES_VEHICLE = ["car", "truck", "bus", "motorcycle"]
CLASSES_ANIMAL  = ["dog", "cat", "horse", "cow", "sheep"]

danger_zone = None

# ── بارگذاری حریم قبلی از فایل JSON ───────────────────────
def load_zone_from_json():
    global danger_zone
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                points = [tuple(p) for p in data.get("danger_zone", [])]
                if len(points) >= 3:
                    danger_zone = np.array(points, np.int32)
                    print(f"✅ حریم تثبیت‌شده از '{CONFIG_FILE}' بارگذاری شد: {points}")
                    return True
        except Exception as e:
            print(f"⚠️ خطا در خواندن فایل حریم: {e}")
    print("❌ فایل zone_config.json یافت نشد!")
    return False

# ═══════════════════════════════════════════════════════════
# 🧠 راه اندازی و شروع برنامه
# ═══════════════════════════════════════════════════════════
print("=" * 70)
print("🚀 سیستم پایش حریم اختصاصی خیابان (نسخه Colab Live Stream)")
print("=" * 70)

if not load_zone_from_json():
    raise FileNotFoundError("لطفاً ابتدا فایل zone_config.json را در مسیر مشخص شده ایجاد کنید.")

model = YOLO("yolov8s.pt")
cap = cv2.VideoCapture(VIDEO_SOURCE)
last_alert_time = 0

# ساخت Handeling اختصاصی برای استریم تصویری در Colab
display_handle = display(None, display_id=True)

def frame_to_b64(frame):
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()

def in_zone(cx, cy, poly):
    return cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0

# ═══════════════════════════════════════════════════════════
# 🔁 حلقه اصلی پردازش
# ═══════════════════════════════════════════════════════════
frame_count = 0

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    now = time.time()
    frame_count += 1

    # رسم محدوده حریم اختصاصی روی فریم
    overlay = frame.copy()
    cv2.fillPoly(overlay, [danger_zone], (0, 0, 180))
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)
    cv2.polylines(frame, [danger_zone], True, (0, 0, 255), 2)

    # اجرای تشخیص YOLO
    results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)
    zone_violations = []

    for r in results:
        if r.boxes is None: continue
        for box in r.boxes:
            cls_name = model.names[int(box.cls[0])]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            if in_zone(cx, cy, danger_zone):
                if cls_name in CLASSES_PERSON:
                    zone_violations.append(("آدم", "Hazard", "Medium"))
                elif cls_name in CLASSES_VEHICLE:
                    zone_violations.append(("خودرو", "Hazard", "High"))
                elif cls_name in CLASSES_ANIMAL:
                    zone_violations.append(("حیوان", "Hazard", "Low"))

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"INTRUSION: {cls_name}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # ارسال حادثه به کنترلر C# (.NET)
    if zone_violations and (now - last_alert_time > COOLDOWN_SECONDS):
        v_type, severity, _ = zone_violations[0]
        title = f"خطر: ورود غیرمجاز {v_type} به حریم اختصاصی"

        payload = {
            "title": title,
            "type": "Hazard",
            "severity": severity,
            "imageUrl": frame_to_b64(frame),
            "location": CAMERA_NAME,
            "department": "حفاظت و ایمنی محیطی",
            "incidentDate": datetime.now().isoformat(),
            "description": f"شناسایی ورود غیرمجاز ({v_type}) به حریم تثبیت‌شده."
        }

        try:
            r = requests.post(API_URL, json=payload, verify=False, timeout=5)
            if r.status_code in (200, 201):
                print(f"🚨 [{datetime.now():%H:%M:%S}] ثبت در API: {title}")
                last_alert_time = now
            else:
                print(f"⚠️ پاسخ API: Status {r.status_code}")
        except Exception as e:
            print(f"❌ خطا در ارسال API: {e}")

    # ── 📺 نمایش زنده در گوگل کولب ────────────────────────
    # برای روان‌تر شدن اجرا، اندازه تصویر را کمی کوچک می‌کنیم
    preview_frame = cv2.resize(frame, (640, 360))
    _, jpeg = cv2.imencode('.jpg', preview_frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
    display_handle.update(Image(data=jpeg.tobytes()))

    time.sleep(0.01)

cap.release()