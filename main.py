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
# ⚙️ تنظیمات
# ═══════════════════════════════════════════════════════════
BASE_NGROK_URL   = "https://outfit-dimly-juice.ngrok-free.dev"
API_URL          = f"{BASE_NGROK_URL}/api/Violations/camera"

CAMERA_NAME      = "دوربین ۲ - خط تولید A"

# 🔴 حتماً مسیر درست فایل ویدیو در گوگل درایو را قرار دهید
VIDEO_SOURCE     = "/content/drive/MyDrive/HSEPlatform/HSE.AI/lifterac.mp4" 

COOLDOWN_SECONDS = 30
CONF_THRESHOLD   = 0.35   # آستانه اطمینان تشخیص
CONFIRM_FRAMES   = 3

print("=" * 70)
print("🚀 بارگذاری مدل اختصاصی HSE (best.pt)...")
print("=" * 70)

# ۱. بررسی وجود فایل مدل
if not os.path.exists("best.pt"):
    # اگر فایل در ریشه اصلی کولب است، آن را کپی کنیم
    if os.path.exists("/content/best.pt"):
        os.system("cp /content/best.pt ./best.pt")
    else:
        print("❌ خطای حیاتی: فایل best.pt یافت نشد!")
        exit(1)

# ۲. بررسی وجود فایل ویدیو
if not os.path.exists(VIDEO_SOURCE):
    print(f"❌ خطای حیاتی: فایل ویدیو در مسیر زیر پیدا نشد:\n📍 {VIDEO_SOURCE}")
    print("💡 لطفاً مسیر ویدیوی موجود در Google Drive را اصلاح کنید.")
    exit(1)

# 🧠 بارگذاری وزن‌های اختصاصی
model = YOLO("best.pt")
cap = cv2.VideoCapture(VIDEO_SOURCE)

if not cap.isOpened():
    print("❌ خطای باز کردن فایل ویدیو!")
    exit(1)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS) or 25
print(f"✅ ویدیو باز شد | کل فریم‌ها: {total_frames} | FPS: {fps:.0f}\n")

last_alert_time     = 0
is_violation_active = False
confirm_counter     = 0
frame_count         = 0

def convert_frame_to_base64(frame, max_width=640):
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / float(w)
        frame = cv2.resize(frame, (max_width, int(h * scale)))
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
    return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

# ═══════════════════════════════════════════════════════════
# 🔁 حلقه اصلی پردازش
# ═══════════════════════════════════════════════════════════
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("\n🏁 پایان ویدیو رسید.")
        break

    frame_count += 1

    # اجرای استنتاج مستقیم با مدل
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

            # تشخیص عدم استفاده از کلاه ایمنی
            if class_name.lower() in ["no helmet", "no_helmet"]:
                current_frame_has_violation = True
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"No Helmet ({conf:.0%})", 
                            (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

    # لاگ پیشرفت در ترمینال کولب
    print(f"🔄 پردازش فریم {frame_count}/{total_frames} | وضعیت تخلف: {current_frame_has_violation}", end="\r")

    # سیستم تأیید چند فریمی
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
                    print(f"\n🎯 [{datetime.now():%H:%M:%S}] تخلف با موفقیت در API ثبت شد!")
                    last_alert_time = current_time
                    is_violation_active = True
                else:
                    print(f"\n⚠️ پاسخ API: {response.status_code}")
            except Exception as e:
                print(f"\n❌ خطا در ارسال به API: {e}")
                last_alert_time = current_time
    else:
        if current_time - last_alert_time > COOLDOWN_SECONDS:
            is_violation_active = False

cap.release()
print("\n✅ پردازش کامل شد.")