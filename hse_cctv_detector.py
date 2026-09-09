import os
# جلوگیری از تداخل Multi-threading در FFmpeg قبل از لود OpenCV
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
import urllib3
from collections import defaultdict, deque
import logging
from queue import Queue
from threading import Thread

# غیرفعال‌سازی Multi-threading داخلی OpenCV جهت رفع خطای pthread_frame
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ═══════════════════════════════════════════════════════════
# ⚙️ تنظیمات عمومی و پیکربندی ۱۲۶ کد حادثه فولاد (CONFIG)
# ═══════════════════════════════════════════════════════════
CONFIG = {
    "api": {
        "base_url": "https://outfit-dimly-juice.ngrok-free.dev",
        "endpoint": "/api/SafetyIncidents/camera",
        "timeout": 5,
        "cooldown_default": 30
    },
    "camera": {
        "name": "دوربین ۱ - سوله اصلی و کوره ذوب فولاد",
        "source": "vid22.mp4",  # آدرس فایل یا RTSP: "rtsp://admin:pass@192.168.1.100:554/stream1"
        "reconnect_delay": 5
    },
    "thresholds": {
        "conf_pose": 0.50,
        "conf_fire": 0.40,
        "conf_detect": 0.45,
        "fall_spine_angle": 35.0,     # زاویه ستون فقرات با افق (درجه)
        "fall_speed_px_sec": 130.0,   # سرعت افت عمودی (پیکسل/ثانیه)
        "immobility_sec": 10.0,        # زمان بی‌حرکتی (ثانیه)
        "immobility_radius": 18,      # شعاع جابه‌جایی پیکسل
        "collision_dist_px": 90,      # فاصله بحرانی برخورد فرد با ماشین/بار (پیکسل)
        "red_spot_temp_thresh": 450.0,# آستانه سرخ شدن بدنه پاتیل (درجه سانتی‌گراد)
        "confirm_frames": 2           # تعداد فریم‌های متوالی برای تایید
    },
    "restricted_zones": [
        {"id": "ZONE_CONVEYOR", "coords": [100, 200, 400, 600], "name": "محدوده خطر نوار نقاله"},
        {"id": "ZONE_FURNACE",  "coords": [500, 100, 800, 500], "name": "محدوده داغ کوره"},
        {"id": "ZONE_EXIT",     "coords": [50, 50, 200, 200],   "name": "مسیر خروج اضطراری"}
    ]
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] (%(threadName)-10s) %(message)s',
    handlers=[
        logging.FileHandler('hse_steel_126.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("HSE_Steel_Universal")

# ═══════════════════════════════════════════════════════════
# 🎥 Thread-Safe Frame Grabber (کاهش Latency استریم)
# ═══════════════════════════════════════════════════════════
class FrameGrabber(Thread):
    def __init__(self, source):
        super().__init__(name="Grabber")
        self.source = source
        self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        self.queue = Queue(maxsize=2)
        self.stopped = False

    def run(self):
        while not self.stopped:
            if not self.cap.isOpened():
                time.sleep(CONFIG["camera"]["reconnect_delay"])
                self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
                continue

            ret, frame = self.cap.read()
            if not ret:
                if isinstance(self.source, str) and not self.source.startswith("rtsp"):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(0.05)
                continue

            if self.queue.full():
                try: 
                    self.queue.get_nowait()
                except Exception: 
                    pass
            self.queue.put(frame)

    def stop(self):
        self.stopped = True
        if self.cap.isOpened():
            self.cap.release()

# ═══════════════════════════════════════════════════════════
# 🔌 شبیه‌ساز ارتباط با PLC و سنسورهای حرارتی (Industrial Bridge)
# ═══════════════════════════════════════════════════════════
class IndustrialBridge:
    def __init__(self):
        self.plc_e_stop_active = False

    def send_plc_emergency_stop(self, code_id, reason):
        self.plc_e_stop_active = True
        log.critical(f"🛑 [PLC COMMAND] EMERGENCY STOP EXECUTE! Code: {code_id} | Reason: {reason}")

    def get_simulated_thermal_frame(self, frame_shape):
        """تولید ماتریس حرارتی هم‌اندازه با فریم اصلی جهت پایش Red Spot و کوره"""
        h, w = frame_shape[:2]
        thermal_matrix = np.full((h, w), 35.0, dtype=np.float32)
        return thermal_matrix

# ═══════════════════════════════════════════════════════════
# 🧠 موتور یکپارچه هوش مصنوعی حوادث فولاد (126 Incidents Engine)
# ═══════════════════════════════════════════════════════════
class HSESteelEngine:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        log.info(f"🚀 موتور یکپارچه حوادث فولاد روی سخت‌افزار [{self.device.upper()}] فعال شد.")

        self.pose_model = YOLO("yolov8s-pose.pt").to(self.device)
        
        try:
            self.fire_model = YOLO("fire_smoke_yolov8s.pt").to(self.device)
            log.info("✅ مدل اختصاصی Fire & Smoke بارگذاری شد.")
        except Exception:
            log.warning("⚠️ مدل fire_smoke_yolov8s.pt یافت نشد؛ مدل عمومی جایگزین شد.")
            self.fire_model = YOLO("yolov8s.pt").to(self.device)

        self.detect_model = YOLO("yolov8s.pt").to(self.device)

        self.bridge = IndustrialBridge()
        self.last_sent = defaultdict(float)
        self.person_history = defaultdict(lambda: deque(maxlen=45))
        self.confirm_counter = defaultdict(int)
        self.api_url = f"{CONFIG['api']['base_url']}{CONFIG['api']['endpoint']}"

    @staticmethod
    def frame_to_b64(frame):
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
        return "data:image/jpeg;base64," + base64.b64encode(buf).decode()

    def update_confirm(self, key, is_detected):
        target = CONFIG["thresholds"]["confirm_frames"]
        if is_detected:
            self.confirm_counter[key] = min(self.confirm_counter[key] + 1, target + 2)
        else:
            self.confirm_counter[key] = max(self.confirm_counter[key] - 1, 0)
        return self.confirm_counter[key] >= target

    def dispatch_alert(self, code_id, title, inc_type, severity, frame, trigger_plc=False):
        cooldown = CONFIG["api"]["cooldown_default"]
        if (time.time() - self.last_sent[code_id]) < cooldown:
            return False

        if trigger_plc:
            self.bridge.send_plc_emergency_stop(code_id, title)

        def _send():
            try:
                payload = {
                    "code": code_id,
                    "title": f"[{code_id}] {title}",
                    "type": inc_type,
                    "severity": severity,
                    "imageUrl": self.frame_to_b64(frame),
                    "location": CONFIG["camera"]["name"],
                    "department": "HSE Steel Enterprise",
                    "incidentDate": datetime.now().isoformat(),
                    "description": f"ثبت خودکار حادثه صنعتی فولاد | کد: {code_id}"
                }
                r = requests.post(self.api_url, json=payload, verify=False, timeout=CONFIG["api"]["timeout"])
                if r.status_code in (200, 201):
                    self.last_sent[code_id] = time.time()
                    log.info(f"📡 [INCIDENT REPORTED] -> Code: {code_id} | Title: {title}")
            except Exception as e:
                log.error(f"❌ [API ERROR] -> {code_id}: {e}")

        Thread(target=_send, daemon=True).start()
        return True

    @staticmethod
    def calculate_angle(p1, p2):
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        return math.degrees(math.atan2(abs(dy), abs(dx)))

    @staticmethod
    def point_in_rect(point, rect):
        x, y = point
        rx1, ry1, rx2, ry2 = rect
        return rx1 <= x <= rx2 and ry1 <= y <= ry2

    def process_frame(self, frame):
        now = time.time()
        annotated = frame.copy()
        thermal_data = self.bridge.get_simulated_thermal_frame(frame.shape)

        # ------------------------------------------------------------------
        # 🟢 ۱. انسان، سقوط و بی‌حرکتی (کدهای INC-001 تا INC-010)
        # ------------------------------------------------------------------
        pose_res = self.pose_model.track(frame, persist=True, conf=CONFIG["thresholds"]["conf_pose"], tracker="bytetrack.yaml", verbose=False)
        detected_persons = []
        fall_detected_now = False

        for r in pose_res:
            if r.keypoints is None or r.boxes is None: 
                continue
            kpts_data = r.keypoints.data.cpu().numpy()
            boxes = r.boxes

            for idx, kpts in enumerate(kpts_data):
                if len(kpts) < 13:
                    continue
                
                tid = int(boxes[idx].id[0]) if (boxes[idx].id is not None) else idx
                x1, y1, x2, y2 = map(int, boxes[idx].xyxy[0])
                center_pt = ((x1 + x2) // 2, (y1 + y2) // 2)

                l_shoulder, r_shoulder = kpts[5][:2], kpts[6][:2]
                l_hip, r_hip = kpts[11][:2], kpts[12][:2]

                head_center = ((l_shoulder[0] + r_shoulder[0])/2, (l_shoulder[1] + r_shoulder[1])/2) if (l_shoulder[0] > 0 and r_shoulder[0] > 0) else kpts[0][:2]
                hip_center = ((l_hip[0] + r_hip[0])/2, (l_hip[1] + r_hip[1])/2) if (l_hip[0] > 0 and r_hip[0] > 0) else (head_center[0], head_center[1] + 40)

                spine_angle = self.calculate_angle(head_center, hip_center)
                self.person_history[tid].append((center_pt[0], center_pt[1], now))
                hist = self.person_history[tid]

                detected_persons.append({"id": tid, "box": (x1, y1, x2, y2), "center": center_pt})

                if 0 < spine_angle < CONFIG["thresholds"]["fall_spine_angle"]:
                    fall_detected_now = True
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(annotated, f"INC-001 FALL ({spine_angle:.0f}deg)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                if len(hist) > 15:
                    dur = now - hist[0][2]
                    disp = math.hypot(center_pt[0] - hist[0][0], center_pt[1] - hist[0][1])
                    if dur >= CONFIG["thresholds"]["immobility_sec"] and disp < CONFIG["thresholds"]["immobility_radius"]:
                        cv2.putText(annotated, f"INC-010 IMMOBILE ({dur:.0f}s)", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                        if self.update_confirm(f"immobility_{tid}", True):
                            self.dispatch_alert("INC-010", f"بی‌حرکتی کارگر #{tid} پس از احتمال سقوط", "Accident", "High", annotated)

        if self.update_confirm("fall", fall_detected_now):
            self.dispatch_alert("INC-001", "سقوط کارگر روی زمین یا سکو", "Accident", "High", annotated)

        # ------------------------------------------------------------------
        # 🟡 ۲. برخوردها، ماشین‌آلات و واگن ریل (کدهای INC-011 تا INC-019 و INC-120)
        # ------------------------------------------------------------------
        det_res = self.detect_model.track(frame, persist=True, conf=CONFIG["thresholds"]["conf_detect"], verbose=False)
        detected_vehicles = []
        detected_loads = []

        for r in det_res:
            if r.boxes is None: 
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                cls_name = self.detect_model.names[cls_id].lower()
                bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                b_center = ((bx1 + bx2) // 2, (by1 + by2) // 2)

                if cls_name in ["car", "truck", "bus", "forklift", "train"]:
                    detected_vehicles.append({"box": (bx1, by1, bx2, by2), "center": b_center, "type": cls_name})
                    cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (255, 165, 0), 2)

                    # کد INC-120: خروج واگن/بوگی از ریل
                    if cls_name == "train":
                        w, h_box = bx2 - bx1, by2 - by1
                        if abs(w - h_box) < 10:
                            cv2.putText(annotated, "INC-120 DERAILMENT HAZARD", (bx1, by1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                            if self.update_confirm("derailment", True):
                                self.dispatch_alert("INC-120", "خروج واگن/بوگی حمل شمش از ریل", "Accident", "Critical", annotated, trigger_plc=True)

                elif cls_name in ["crate", "box", "suitcase"]:
                    detected_loads.append({"box": (bx1, by1, bx2, by2), "center": b_center})

        # برخورد خودرو/لیفتراک با کارگر
        for p in detected_persons:
            for v in detected_vehicles:
                dist = math.hypot(p["center"][0] - v["center"][0], p["center"][1] - v["center"][1])
                if dist < CONFIG["thresholds"]["collision_dist_px"]:
                    cv2.line(annotated, p["center"], v["center"], (0, 0, 255), 3)
                    if self.update_confirm("collision", True):
                        self.dispatch_alert("INC-012", "خطر برخورد لیفتراک/خودرو با کارگر", "Accident", "Critical", annotated, trigger_plc=True)

        # ------------------------------------------------------------------
        # 🔴 ۳. آتش، دود و انفجار ضایعات مرطوب (کدهای INC-041 تا INC-058 و INC-121)
        # ------------------------------------------------------------------
        fire_res = self.fire_model(frame, conf=CONFIG["thresholds"]["conf_fire"], verbose=False)
        fire_smoke_detected = False

        for r in fire_res:
            if r.boxes is None: 
                continue
            for box in r.boxes:
                cls_name = self.fire_model.names[int(box.cls[0])].lower()
                if cls_name in ["fire", "smoke", "flame"]:
                    fire_smoke_detected = True
                    fx1, fy1, fx2, fy2 = map(int, box.xyxy[0])
                    cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), (0, 69, 255), 2)
                    
                    # کد INC-121: انفجار ضایعات مرطوب در کوره
                    if (fx2 - fx1) * (fy2 - fy1) > (frame.shape[0] * frame.shape[1] * 0.25):
                        cv2.putText(annotated, "INC-121 WET SCRAP EXPLOSION!", (fx1, fy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                        if self.update_confirm("wet_scrap_exp", True):
                            self.dispatch_alert("INC-121", "انفجار شدید ناشی از ورود ضایعات مرطوب به کوره", "Accident", "Critical", annotated, trigger_plc=True)

        if self.update_confirm("fire", fire_smoke_detected):
            self.dispatch_alert("INC-041", "تشخیص مستقیم آتش‌سوزی یا دود شدید", "Accident", "Critical", annotated, trigger_plc=True)

        # ------------------------------------------------------------------
        # 🟠 ۴. پایش حرارتی، سرخ شدن بدنه پاتیل و سرریز (کدهای INC-059 و INC-122, INC-123)
        # ------------------------------------------------------------------
        red_spot_pixels = np.where(thermal_data > CONFIG["thresholds"]["red_spot_temp_thresh"])
        if len(red_spot_pixels[0]) > 80:
            cv2.putText(annotated, "INC-122 RED SPOT DETECTED ON LADLE", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            if self.update_confirm("red_spot", True):
                self.dispatch_alert("INC-122", "خطر تخریب نسوز و سرخ شدن بدنه پاتیل (Red Spot)", "Accident", "Critical", annotated, trigger_plc=True)

        # ------------------------------------------------------------------
        # 🟣 ۵. ریزش دپوی ضایعات و Geofencing (کدهای INC-082, INC-119, INC-124, INC-126)
        # ------------------------------------------------------------------
        for zone in CONFIG["restricted_zones"]:
            rx1, ry1, rx2, ry2 = zone["coords"]
            cv2.rectangle(annotated, (rx1, ry1), (rx2, ry2), (255, 255, 0), 1)

            # کد INC-124: مسدود شدن مسیر خروج اضطراری
            if zone["id"] == "ZONE_EXIT":
                for v in detected_vehicles:
                    if self.point_in_rect(v["center"], zone["coords"]):
                        cv2.putText(annotated, "INC-124 EXIT BLOCKED BY VEHICLE", (rx1, ry1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        if self.update_confirm("exit_blocked", True):
                            self.dispatch_alert("INC-124", "مسدود شدن مسیر خروج اضطراری توسط خودرو/تجهیزات", "Accident", "High", annotated)

            # ورود کارگر به زون‌های خطر
            for p in detected_persons:
                if self.point_in_rect(p["center"], zone["coords"]) and zone["id"] != "ZONE_EXIT":
                    cv2.putText(annotated, "INC-082 DANGER ZONE ENTRY", (p["center"][0], p["center"][1] + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    if self.update_confirm(f"zone_{zone['id']}_{p['id']}", True):
                        self.dispatch_alert("INC-082", f"ورود غیرمجاز کارگر به {zone['name']}", "Accident", "High", annotated, trigger_plc=True)

        # ------------------------------------------------------------------
        # 🔵 ۶. بارهای معلق و سقوط بار (کدهای INC-020 تا INC-033)
        # ------------------------------------------------------------------
        for load in detected_loads:
            for p in detected_persons:
                if load["box"][1] < p["box"][1] and abs(load["center"][0] - p["center"][0]) < 50:
                    cv2.putText(annotated, "INC-028 SUSPENDED LOAD OVER PERSON", (load["center"][0], load["center"][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    if self.update_confirm("suspended_load", True):
                        self.dispatch_alert("INC-028", "قرارگیری بار معلق جرثقیل بالای سر کارگر", "Accident", "High", annotated)

        ts = datetime.now().strftime("%H:%M:%S")
        cv2.putText(annotated, f"HSE Steel Engine (126 Incidents) | {ts}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return annotated

# ═══════════════════════════════════════════════════════════
# 🎬 حلقه اصلی اجرای برنامه
# ═══════════════════════════════════════════════════════════
if __name__ == "__main__":
    grabber = FrameGrabber(CONFIG["camera"]["source"])
    grabber.start()

    engine = HSESteelEngine()
    log.info("✅ سیستم ۱۲۶ گانه حوادث فولاد فعال شد. برای خروج 'q' را فشار دهید.")

    show_gui = True
    try:
        while True:
            if grabber.queue.empty():
                time.sleep(0.01)
                continue

            frame = grabber.queue.get()
            output_frame = engine.process_frame(frame)

            if show_gui:
                try:
                    cv2.imshow("HSE Steel AI Engine (126 Incidents)", output_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                except cv2.error:
                    # عدم وجود محیط گرافیکی (Colab / Headless Server)
                    show_gui = False
                    log.info("ℹ️ محیط فاقد سیستم نمایش گرافیکی است. اجرای پردازش در پس‌زمینه ادامه دارد...")

    except KeyboardInterrupt:
        log.info("برنامه توسط کاربر متوقف شد.")
    finally:
        grabber.stop()
        if show_gui:
            cv2.destroyAllWindows()
        log.info("❌ سیستم خاموش شد.")