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

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ═══════════════════════════════════════════════════════════
# ⚙️ تنظیمات اصلی
# ═══════════════════════════════════════════════════════════
BASE_NGROK_URL   = "https://outfit-dimly-juice.ngrok-free.dev"
API_URL          = f"{BASE_NGROK_URL}/api/SafetyIncidents/camera"

CAMERA_NAME      = "دوربین حریم اختصاصی - street"
VIDEO_SOURCE     = "street.mp4"
CONFIG_FILE      = "zone_config.json"  # فایل ذخیره‌سازی دائمی مختصات حریم
COOLDOWN_SECONDS = 15
CONF_THRESHOLD   = 0.40

CLASSES_PERSON  = ["person"]
CLASSES_VEHICLE = ["car", "truck", "bus", "motorcycle"]
CLASSES_ANIMAL  = ["dog", "cat", "horse", "cow", "sheep"]

# ── متغیرهای وضعیت رسم حریم ──────────────────────────────
drawn_points = []
danger_zone = None
drawing_complete = False

# ── توابع مدیریت فایل پیکربندی (Save & Load) ──────────────
def save_zone_to_json(points):
    """ذخیره دائمی نقاط حریم در فایل JSON"""
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump({"danger_zone": points}, f, indent=4)
        print(f"💾 حریم با موفقیت در فایل '{CONFIG_FILE}' ذخیره شد.")
    except Exception as e:
        print(f"❌ خطا در ذخیره‌سازی فایل حریم: {e}")

def load_zone_from_json():
    """بارگذاری حریم قبلی از فایل JSON در صورت وجود"""
    global drawn_points, danger_zone, drawing_complete
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                drawn_points = [tuple(p) for p in data.get("danger_zone", [])]
                if len(drawn_points) >= 3:
                    danger_zone = np.array(drawn_points, np.int32)
                    drawing_complete = True
                    print(f"✅ حریم تثبیت‌شده قبلی از '{CONFIG_FILE}' با موفقیت بارگذاری شد.")
        except Exception as e:
            print(f"⚠️ خطا در خواندن فایل حریم: {e}")

def mouse_callback(event, x, y, flags, param):
    """مدیریت تعاملی ماوس جهت تعیین نقاط"""
    global drawn_points, danger_zone, drawing_complete

    if event == cv2.EVENT_LBUTTONDOWN and not drawing_complete:
        drawn_points.append((x, y))
        print(f"📍 نقطه اضافه شد: ({x}, {y})")

    elif event == cv2.EVENT_RBUTTONDOWN and not drawing_complete:
        if len(drawn_points) >= 3:
            danger_zone = np.array(drawn_points, np.int32)
            drawing_complete = True
            save_zone_to_json(drawn_points)  # ذخیره خودکار در فایل
        else:
            print("⚠️ برای ایجاد حریم حداقل به ۳ یا ۴ نقطه نیاز است!")

# ═══════════════════════════════════════════════════════════
# 🧠 راه اندازی و شروع برنامه
# ═══════════════════════════════════════════════════════════
print("=" * 70)
print("🚀 سیستم پایش حریم اختصاصی (با قابلیت ذخیره همیشگی)")
print("  • اگر قبلاً حریم کشیده باشید، به‌صورت خودکار بارگذاری می‌شود.")
print("  • برای رسم حریم جدید: کلیک چپ (افزودن نقطه) -> کلیک راست (ثبت و ذخیره همیشگی)")
print("  • برای پاک کردن حریم تثبیت‌شده و رسم مجدد: کلید 'r' را روی کیبورد بزنید.")
print("=" * 70)

# بارگذاری حریم ثبت شده قبلی
load_zone_from_json()

model = YOLO("yolov8s.pt")
cap = cv2.VideoCapture(VIDEO_SOURCE)

cv2.namedWindow("HSE Street Camera")
cv2.setMouseCallback("HSE Street Camera", mouse_callback)

last_alert_time = 0

def frame_to_b64(frame):
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()

def in_zone(cx, cy, poly):
    return cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0

# ═══════════════════════════════════════════════════════════
# 🔁 حلقه اصلی پردازش
# ═══════════════════════════════════════════════════════════
while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    now = time.time()

    # ── ۱. حالت اول: کاربر در حال رسم حریم جدید است ──────────
    if not drawing_complete:
        for pt in drawn_points:
            cv2.circle(frame, pt, 5, (0, 255, 255), -1)
        if len(drawn_points) > 1:
            cv2.polylines(frame, [np.array(drawn_points, np.int32)], False, (0, 255, 255), 2)

        cv2.putText(frame, "Click points then RIGHT-CLICK to lock & save forever", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

    # ── ۲. حالت دوم: حریم قفل شده و پایش فعال است ─────────────
    else:
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
            except Exception as e:
                print(f"❌ خطا در ارسال API: {e}")

    # راهنما و نمایش ویدیو
    cv2.putText(frame, "Press 'r' to Reset Zone | 'q' to Quit", (10, frame.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("HSE Street Camera", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('r'):
        # ریست کامل حریم و حذف فایل ذخیره‌شده
        drawn_points.clear()
        danger_zone = None
        drawing_complete = False
        if os.path.exists(CONFIG_FILE):
            os.remove(CONFIG_FILE)
        print("🔄 فایل پیکربندی حذف شد. می‌توانید حریم جدیدی بکشید.")

    time.sleep(0.03)

cap.release()
cv2.destroyAllWindows()