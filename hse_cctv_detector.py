%%writefile hse_cctv_detector.py
import cv2
import numpy as np
from ultralytics import YOLO
import requests
import time
import math
import base64
import os
import sys
from datetime import datetime
import urllib3

# غیرفعال کردن هشدارهای SSL
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ═══════════════════════════════════════════════════════════
# ⚙️ تنظیمات اصلی و متغیرها
# 🎯 آدرس ngrok سیستم خود را جایگزین کنید!
# ═══════════════════════════════════════════════════════════

BASE_NGROK_URL   = "https://a1b2-c3d4-e5f6.ngrok-free.app"  # آدرس اختصاصی Ngrok شما
API_URL          = f"{BASE_NGROK_URL}/api/safetyincidents/camera"
CAMERA_NAME      = "دوربین ۱ - سوله اصلی HSE"
COOLDOWN_SECONDS = 15     # فاصله زمانی بین دو ارسال هشدار یکسان
CONF_THRESHOLD   = 0.55   # حداقل درصد اطمینان مدل برای تشخیص (کاهش خطای مثبت کاذب)

# فایل ویدیو یا آدرس استریم RTSP
VIDEO_SOURCE = "vid2.mp4"

def start_alarm():
    """هشدار متنی در ترمینال"""
    print("\a🚨 [آژیر خطر HSE فعال شد!]", end="\r")

def stop_alarm():
    pass

def frame_to_b64(frame):
    """تبدیل فریم به فرمت Base64 برای ارسال به دات‌نت"""
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode('utf-8')

def in_zone(x, y, poly):
    """بررسی قرارگیری نقطه در منطقه خطر"""
    return cv2.pointPolygonTest(poly, (float(x), float(y)), False) >= 0

# ═══════════════════════════════════════════════════════════
# 🧠 بارگذاری مدل‌های هوشمند YOLO
# ═══════════════════════════════════════════════════════════
print("=" * 75)
print("🚀 سیستم ارتقایافته تشخیص حوادث صنعتی HSE با YOLOv8-Pose فعال شد...")
print("=" * 75)

# مدل اصلی برای شناسایی اشیاء و وسایل نقلیه (Medium برای دقت بالا)
det_model = YOLO("yolov8m.pt") 

# مدل تخصصی تشخیص مفاصل و حالت بدن برای تشخیص دقیق سقوط
pose_model = YOLO("yolov8m-pose.pt")

cap = cv2.VideoCapture(VIDEO_SOURCE)

# ─── وضعیت‌های پایش (State Management) ────────────────────
last_alert_time       = 0
active_incident_title = None
immobility_tracker    = {}
fall_frame_counter    = 0   # بافر زمانی برای تایید سقوط (تکرار ۵ فریم متوالی)

# ═══════════════════════════════════════════════════════════
# 🔁 حلقه اصلی پردازش ویدیویی
# ═══════════════════════════════════════════════════════════
while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        immobility_tracker.clear()
        fall_frame_counter = 0
        active_incident_title = None
        continue

    h, w = frame.shape[:2]
    now  = time.time()

    # ── ۱. تعریف و رسم حریم خطرناک ─────────────────────────
    danger_zone = np.array([
        [int(w * 0.05), int(h * 0.35)],
        [int(w * 0.45), int(h * 0.35)],
        [int(w * 0.45), int(h * 0.90)],
        [int(w * 0.05), int(h * 0.90)]
    ], np.int32)

    overlay = frame.copy()
    cv2.fillPoly(overlay, [danger_zone], (0, 0, 180))
    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
    cv2.polylines(frame, [danger_zone], True, (0, 0, 255), 2)

    # ── ۲. استخراج مفاصل و تشخیص سقوط پیشرفته (YOLO Pose) ───
    pose_results = pose_model(frame, conf=CONF_THRESHOLD, verbose=False)
    raw_fall_detected = False

    for r in pose_results:
        if r.keypoints is not None and len(r.keypoints) > 0:
            for kpts in r.keypoints.data:
                # خروجی مفاصل: شانه چپ (۵)، شانه راست (۶)، لگن چپ (۱۱)، لگن راست (۱۲)
                if len(kpts) >= 13:
                    l_shoulder, r_shoulder = kpts[5][:2], kpts[6][:2]
                    l_hip, r_hip           = kpts[11][:2], kpts[12][:2]

                    # بررسی میانگین مختصات شانه و لگن
                    if l_shoulder[0] > 0 and l_hip[0] > 0:
                        shoulder_y = (l_shoulder[1] + r_shoulder[1]) / 2.0
                        hip_y      = (l_hip[1] + r_hip[1]) / 2.0
                        shoulder_x = (l_shoulder[0] + r_shoulder[0]) / 2.0
                        hip_x      = (l_hip[0] + r_hip[0]) / 2.0

                        dx = abs(shoulder_x - hip_x)
                        dy = abs(shoulder_y - hip_y)

                        # محاسبه زاویه بدن نسبت به افق
                        if dy > 0:
                            angle = math.degrees(math.atan(dx / dy))
                            # اگر زاویه بدن با افق بیش از ۶۰ درجه متمایل شد = سقوط
                            if angle > 60 or shoulder_y > hip_y:
                                raw_fall_detected = True
                                break

    # بافر زمانی سقوط برای حذف نویز
    if raw_fall_detected:
        fall_frame_counter += 1
    else:
        fall_frame_counter = max(0, fall_frame_counter - 1)

    # ── ۳. اجرای مدل تشخیص اشیا و وسایل نقلیه ─────────────
    det_results = det_model(frame, conf=CONF_THRESHOLD, verbose=False)

    persons  = []
    vehicles = []

    for r in det_results:
        for box in r.boxes:
            cls_name = det_model.names[int(box.cls[0])]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            if cls_name == "person":
                persons.append({'box': (x1, y1, x2, y2), 'center': (cx, cy)})
            elif cls_name in ("car", "truck", "bus", "forklift"):
                vehicles.append({'box': (x1, y1, x2, y2), 'center': (cx, cy)})

    # ── ۴. منطق اولویت‌بندی حوادث ──────────────────────────
    detected     = False
    inc_title    = ""
    inc_type     = "Accident"
    inc_severity = "High"

    # قانون ۱: سقوط تایید شده (تکرار ۵ فریم پشت هم)
    if fall_frame_counter >= 5:
        detected, inc_title = True, "حادثه: سقوط و زمین‌خوردن کارگر"

    # قانون ۲: ورود غیرمجاز به حریم خطر
    if not detected:
        for p in persons:
            cx, cy = p['center']
            if in_zone(cx, cy, danger_zone):
                detected, inc_title, inc_type = True, "خطر: ورود غیرمجاز به حریم خطرناک", "Hazard"
                break

    # قانون ۳: برخورد لیفتراک/خودرو با کارگر
    if not detected:
        for p in persons:
            px, py = p['center']
            for v in vehicles:
                vx, vy = v['center']
                if math.hypot(px - vx, py - vy) < 130:
                    detected, inc_title, inc_type = True, "خطر: احتمال برخورد لیفتراک با کارگر", "NearMiss"
                    break

    # قانون ۴: بی‌حرکتی طولانی (احتمال بیهوشی/حادثه)
    if not detected:
        for idx, p in enumerate(persons):
            cx, cy = p['center']
            key    = f"p_{idx}"
            if key not in immobility_tracker:
                immobility_tracker[key] = (cx, cy, now)
            else:
                ox, oy, st = immobility_tracker[key]
                if math.hypot(cx - ox, cy - oy) < 12:
                    dur = now - st
                    if dur > 6.0:  # اگر بیش از ۶ ثانیه کاملا ثابت ماند
                        detected, inc_title = True, "حادثه: بی‌حرکتی / احتمال بیهوشی کارگر"
                        break
                else:
                    immobility_tracker[key] = (cx, cy, now)

    # ── ۵. ارسال گزارش حادثه به API دات‌نت ──────────────────
    if detected:
        start_alarm()
        if (active_incident_title != inc_title and now - last_alert_time > COOLDOWN_SECONDS):
            try:
                payload = {
                    "title":        inc_title,
                    "type":         inc_type,
                    "severity":     inc_severity,
                    "imageUrl":     frame_to_b64(frame),
                    "location":     CAMERA_NAME,
                    "department":   "ایمنی و HSE",
                    "incidentDate": datetime.now().isoformat(),
                    "description":  f"شناسایی هوشمند با مدل Pose: {inc_title}"
                }
                r = requests.post(API_URL, json=payload, verify=False, timeout=5)
                if r.status_code in (200, 201):
                    print(f"✅ [{datetime.now():%H:%M:%S}] ثبت شد در API: {inc_title}")
                    last_alert_time       = now
                    active_incident_title = inc_title
                else:
                    print(f"⚠️ پاسخ API: status={r.status_code}")
            except Exception as e:
                print(f"❌ خطا در ارسال به API: {e}")
    else:
        if now - last_alert_time > COOLDOWN_SECONDS:
            active_incident_title = None
            stop_alarm()

    time.sleep(0.02)

cap.release()