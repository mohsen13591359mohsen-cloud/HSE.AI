import os
# غیرفعال‌سازی Multi-threading در FFmpeg جهت جلوگیری از کرش
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "threads;1"

import cv2
import numpy as np
import torch
from ultralytics import YOLO
import requests
import time
import math
import base64
from datetime import datetime
from collections import defaultdict, deque
from threading import Thread
import logging

# بهینه‌سازی سرعت OpenCV
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

# ═══════════════════════════════════════════════════════════
# ⚙️ تنظیمات و آستانه‌های دقیق (CONFIG)
# ═══════════════════════════════════════════════════════════
CONFIG = {
    "source": "vid22.mp4",  # آدرس فایل یا استریم RTSP دوربین
    "api_url": "https://outfit-dimly-juice.ngrok-free.dev/api/SafetyIncidents/camera",
    "cooldown_sec": 15,     # زمان انتظار بین ارسال دو هشدار همسان (ثانیه)
    "thresholds": {
        "fall_speed_px_sec": 160.0,   # حداقل سرعت سقوط عمودی (پیکسل/ثانیه)
        "spine_angle_horizon": 35.0,  # زاویه ستون فقرات با سطح افق (کمتر از ۳۵ درجه یعنی بدن افقی شده)
        "aspect_ratio_fall": 1.1,     # نسبت عرض به ارتفاع باکس (در سقوط W/H بیشتر از ۱ می‌شود)
        "fall_confirm_frames": 3,     # تعداد فریم متوالی جهت تایید قطعی سقوط (کاهش هشدار کاذب)
        
        "fire_conf": 0.45,            # آستانه اطمینان مدل آتش
        "fire_growth_rate": 1.35,     # نرخ رشد سریع مساحت آتش/دود در ۳ فریم
        "fire_confirm_frames": 2      # فریم‌های متوالی برای تایید آتش‌سوزی
    }
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [%(levelname)s] - %(message)s')
log = logging.getLogger("Precision_Engine")

# ═══════════════════════════════════════════════════════════
# 🧠 موتور هوشمند سقوط و آتش‌سوزی با دقت بالا
# ═══════════════════════════════════════════════════════════
class HighPrecisionAccidentEngine:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        log.info(f"⚡ اجرای موتور پردازش روی سخت‌افزار: [{self.device.upper()}]")

        # بارگذاری مدل Pose با دقت بالا (Medium Pose)
        self.pose_model = YOLO("yolov8m-pose.pt").to(self.device)
        
        # بارگذاری مدل اختصاصی آتش و دود (در صورت عدم وجود، مدل پایه جایگزین می‌شود)
        try:
            self.fire_model = YOLO("fire_smoke_yolov8s.pt").to(self.device)
            log.info("✅ مدل اختصاصی Fire & Smoke بارگذاری شد.")
        except Exception:
            log.warning("⚠️ مدل اختصاصی آتش یافت نشد. از مدل عمومی yolov8m.pt استفاده می‌شود.")
            self.fire_model = YOLO("yolov8m.pt").to(self.device)

        # ساختار ردیابی فریم به فریم
        self.track_history = defaultdict(lambda: deque(maxlen=20))
        self.fall_confirm_counter = defaultdict(int)
        self.fire_confirm_counter = 0
        self.last_alert_time = defaultdict(float)

    @staticmethod
    def calculate_spine_angle(shoulder, hip):
        """محاسبه زاویه ستون فقرات نسبت به خط افق"""
        dx = hip[0] - shoulder[0]
        dy = hip[1] - shoulder[1]
        return math.degrees(math.atan2(abs(dy), abs(dx) + 1e-5))

    def send_alert(self, code, title, severity, frame):
        """ارسال مجزا و غیرهمگام هشدار به API جهت جلوگیری از افت FPS"""
        if time.time() - self.last_alert_time[code] < CONFIG["cooldown_sec"]:
            return

        self.last_alert_time[code] = time.time()

        def _async_send():
            try:
                _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                b64_img = "data:image/jpeg;base64," + base64.b64encode(buf).decode()

                payload = {
                    "code": code,
                    "title": f"🚨 {title}",
                    "severity": severity,
                    "imageUrl": b64_img,
                    "incidentDate": datetime.now().isoformat(),
                    "description": f"تشخیص هوشمند حادثه با دقت بالا | کد: {code}"
                }
                requests.post(CONFIG["api_url"], json=payload, timeout=4, verify=False)
                log.warning(f"📡 [هشدار ارسال شد] Code: {code} | Title: {title}")
            except Exception as e:
                log.error(f"خطا در ارسال API: {e}")

        Thread(target=_async_send, daemon=True).start()

    def process_frame(self, frame):
        now = time.time()
        annotated = frame.copy()

        # ------------------------------------------------------------------
        # 1️⃣ تشخیص دقیق سقوط کارگر (Fall Detection)
        # ------------------------------------------------------------------
        pose_res = self.pose_model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)

        if pose_res and pose_res[0].keypoints is not None and pose_res[0].boxes is not None:
            boxes = pose_res[0].boxes
            kpts_all = pose_res[0].keypoints.data.cpu().numpy()

            for i, kpts in enumerate(kpts_all):
                if len(kpts) < 13:
                    continue

                tid = int(boxes[i].id[0]) if boxes[i].id is not None else i
                x1, y1, x2, y2 = map(int, boxes[i].xyxy[0])
                
                width = x2 - x1
                height = y2 - y1
                aspect_ratio = width / float(height + 1e-5)
                center_y = (y1 + y2) / 2.0

                # نقاط کلیدی شانه و لگن
                l_shoulder, r_shoulder = kpts[5][:2], kpts[6][:2]
                l_hip, r_hip = kpts[11][:2], kpts[12][:2]

                # محاسبه مرکز شانه و مرکز لگن
                if l_shoulder[0] > 0 and r_shoulder[0] > 0:
                    shoulder_c = ((l_shoulder[0] + r_shoulder[0]) / 2, (l_shoulder[1] + r_shoulder[1]) / 2)
                else:
                    shoulder_c = (x1 + width / 2, y1)

                if l_hip[0] > 0 and r_hip[0] > 0:
                    hip_c = ((l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2)
                else:
                    hip_c = (x1 + width / 2, y2)

                spine_angle = self.calculate_spine_angle(shoulder_c, hip_c)

                # ذخیره موقعیت در تاریخچه
                hist = self.track_history[f"person_{tid}"]
                hist.append((center_y, spine_angle, now))

                is_fall_candidate = False

                if len(hist) >= 4:
                    dt = now - hist[0][2]
                    if dt > 0:
                        # سرعت عمودی جابه‌جایی مرکز بدن (پیکسل بر ثانیه)
                        v_y = (center_y - hist[0][0]) / dt

                        # ۳ شرط همزمان برای تایید اولیه سقوط:
                        if (v_y > CONFIG["thresholds"]["fall_speed_px_sec"] and 
                            spine_angle < CONFIG["thresholds"]["spine_angle_horizon"] and 
                            aspect_ratio > CONFIG["thresholds"]["aspect_ratio_fall"]):
                            is_fall_candidate = True

                # تایید چند فریمی برای حذف هشدارهای اشتباه
                if is_fall_candidate:
                    self.fall_confirm_counter[tid] += 1
                else:
                    self.fall_confirm_counter[tid] = max(0, self.fall_confirm_counter[tid] - 1)

                if self.fall_confirm_counter[tid] >= CONFIG["thresholds"]["fall_confirm_frames"]:
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(annotated, f"🚨 FALL DETECTED (ID: {tid})", (x1, y1 - 12), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    self.send_alert("ACC-FALL", f"سقوط شدید کارگر کد #{tid}", "Critical", annotated)

        # ------------------------------------------------------------------
        # 2️⃣ تشخیص دقیق آتش‌سوزی و دود (Fire & Smoke Detection)
        # ------------------------------------------------------------------
        fire_res = self.fire_model(frame, conf=CONFIG["thresholds"]["fire_conf"], verbose=False)
        fire_detected_in_frame = False

        if fire_res and fire_res[0].boxes is not None:
            for box in fire_res[0].boxes:
                cls_id = int(box.cls[0])
                cls_name = self.fire_model.names[cls_id].lower()

                # بررسی کلاس‌های مرتبط با آتش/دود
                if cls_name in ["fire", "smoke", "flame"]:
                    fx1, fy1, fx2, fy2 = map(int, box.xyxy[0])
                    area = (fx2 - fx1) * (fy2 - fy1)

                    # تحلیل نرخ رشد مساحت در تاریخچه
                    fire_hist = self.track_history["fire_area"]
                    fire_hist.append((area, now))

                    growth_valid = True
                    if len(fire_hist) >= 3:
                        prev_area = fire_hist[0][0]
                        if (area / float(prev_area + 1e-5)) < CONFIG["thresholds"]["fire_growth_rate"] and area < 1500:
                            growth_valid = False

                    if growth_valid:
                        fire_detected_in_frame = True
                        cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), (0, 69, 255), 3)
                        cv2.putText(annotated, f"🔥 {cls_name.upper()} DETECTED", (fx1, fy1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 69, 255), 2)

        # ارزیابی تاییدیه چند فریمی آتش
        if fire_detected_in_frame:
            self.fire_confirm_counter += 1
        else:
            self.fire_confirm_counter = max(0, self.fire_confirm_counter - 1)

        if self.fire_confirm_counter >= CONFIG["thresholds"]["fire_confirm_frames"]:
            self.send_alert("ACC-FIRE", "کشف آتش‌سوزی یا زبانه کشیدن دود", "Critical", annotated)

        # نمایش زمان و عنوان روی تصویر
        cv2.putText(annotated, f"HSE High-Precision Engine | {datetime.now().strftime('%H:%M:%S')}", 
                    (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return annotated

# ═══════════════════════════════════════════════════════════
# 🎬 حلقه اصلی پردازش ویدیو (متقاطع برای Colab و سیستم محلی)
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    cap = cv2.VideoCapture(CONFIG["source"])
    engine = HighPrecisionAccidentEngine()

    log.info("✅ سیستم هوشمند آماده به‌کار شد.")

    # تشخیص محیط Colab یا Headless
    is_headless = "COLAB_GPU" in os.environ or "BUILD_PROP" in os.environ or os.environ.get("DISPLAY") is None

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            log.info("اتمام فایل ویدیو.")
            break

        processed_frame = engine.process_frame(frame)
        frame_count += 1

        # تنها در محیط محلی پنجره گرافیکی باز می‌شود
        if not is_headless:
            cv2.imshow("Precision Fire & Fall Detector", processed_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            if frame_count % 100 == 0:
                log.info(f"فریم‌های پردازش‌شده در Colab: {frame_count}")

    cap.release()
    if not is_headless:
        cv2.destroyAllWindows()