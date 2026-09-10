import os
import time
import math
import base64
import logging
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import Event, Lock

import cv2
import numpy as np
import requests
import torch
from ultralytics import YOLO

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "threads;1")
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

CONFIG = {
    "source": os.getenv("VIDEO_SOURCE", "vid22.mp4"),
    "output": os.getenv("OUTPUT_VIDEO", "hse_annotated.mp4"),
    "process_every_n_frames": int(os.getenv("PROCESS_EVERY_N", "1")),
    "show_gui": os.getenv("SHOW_GUI", "true").lower() == "true",
    "send_alerts": os.getenv("SEND_ALERTS", "false").lower() == "true",
    "api_url": os.getenv("SAFETY_API_URL", "https://outfit-dimly-juice.ngrok-free.dev/api/SafetyIncidents/camera"),
    "verify_tls": os.getenv("VERIFY_TLS", "true").lower() == "true",
    "camera_name": "دوربین ۱ - سوله اصلی و کوره ذوب فولاد",
    "api_timeout": 5,
    "cooldown_sec": 30,
    "thresholds": {
        "pose_conf": 0.55,
        "pose_kpt_conf": 0.50,
        "detect_conf": 0.55,
        "fire_conf": 0.60,
        "fall_speed_px_sec": 130.0,
        "fall_angle_deg": 35.0,
        "fall_aspect_ratio": 1.10,
        "fall_confirm_frames": 4,
        "recovery_frames": 10,
        "immobility_sec": 10.0,
        "immobility_radius_px": 22.0,
        "immobility_confirm_frames": 5,
        "collision_dist_px": 70.0,
        "fire_confirm_frames": 4,
        "fire_recovery_frames": 10,
        "fire_min_area_px": 300,
        "fire_min_area_ratio": 0.001,
        "max_track_age_sec": 5.0,
    },
    # برای هر دوربین باید با مختصات همان تصویر کالیبره شود.
    "restricted_zones": [
        {"id": "ZONE_CONVEYOR", "coords": [100, 200, 400, 600], "name": "محدوده خطر نوار نقاله"},
        {"id": "ZONE_FURNACE", "coords": [500, 100, 800, 500], "name": "محدوده داغ کوره"},
        {"id": "ZONE_EXIT", "coords": [50, 50, 200, 200], "name": "مسیر خروج اضطراری"},
    ],
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("HSE-Steel-Fixed")


class LatestFrameGrabber:
    """برای فایل، ویدیو را یک‌بار می‌خواند؛ برای RTSP با reconnect کار می‌کند."""
    def __init__(self, source):
        self.source = source
        self.is_stream = isinstance(source, str) and source.lower().startswith(("rtsp://", "http://", "https://"))
        self.stop_event = Event()
        self.cap = None

    def frames(self):
        while not self.stop_event.is_set():
            self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            if not self.cap.isOpened():
                self.cap.release()
                if not self.is_stream:
                    raise RuntimeError(f"بازکردن ویدیو شکست خورد: {self.source}")
                log.warning("اتصال دوربین شکست خورد؛ تلاش مجدد...")
                time.sleep(5)
                continue
            while not self.stop_event.is_set():
                ok, frame = self.cap.read()
                if not ok:
                    if self.is_stream:
                        break
                    return
                yield frame
            self.cap.release()
            if self.is_stream:
                time.sleep(5)

    def stop(self):
        self.stop_event.set()
        if self.cap is not None:
            self.cap.release()


class HSESteelEngine:
    def __init__(self, fps):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.fps = max(float(fps), 1.0)
        log.info("دستگاه پردازش: %s | FPS مرجع: %.2f", self.device.upper(), self.fps)
        self.pose_model = YOLO(os.getenv("POSE_MODEL", "yolov8s-pose.pt")).to(self.device)
        fire_path = os.getenv("FIRE_MODEL_PATH", "fire_smoke_yolov8s.pt")
        if not os.path.exists(fire_path):
            raise FileNotFoundError(
                f"مدل اختصاصی آتش یافت نشد: {fire_path}. مدل عمومی COCO برای fire/smoke معتبر نیست."
            )
        self.fire_model = YOLO(fire_path).to(self.device)
        self.detect_model = YOLO(os.getenv("DETECT_MODEL", "yolov8s.pt")).to(self.device)

        self.person_history = defaultdict(lambda: deque(maxlen=max(30, int(self.fps * 15))))
        self.person_state = defaultdict(lambda: {"fall": 0, "immobile": 0, "alerted": False, "last_seen": 0.0})
        self.fire_history = deque(maxlen=max(10, int(self.fps * 3)))
        self.fire_state = {"confirm": 0, "recovery": 0, "alerted": False}
        self.confirm_counter = defaultdict(int)
        self.last_alert = defaultdict(float)
        self.alert_lock = Lock()
        self.sender = ThreadPoolExecutor(max_workers=2)

    @staticmethod
    def keypoint_center(kpts, a, b, minimum):
        if kpts.shape[1] < 3 or kpts[a][2] < minimum or kpts[b][2] < minimum:
            return None
        return ((float(kpts[a][0]) + float(kpts[b][0])) / 2, (float(kpts[a][1]) + float(kpts[b][1])) / 2)

    @staticmethod
    def angle_to_horizontal(a, b):
        return math.degrees(math.atan2(abs(b[1] - a[1]), abs(b[0] - a[0]) + 1e-6))

    @staticmethod
    def box_gap(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        dx = max(ax1 - bx2, bx1 - ax2, 0)
        dy = max(ay1 - by2, by1 - ay2, 0)
        return math.hypot(dx, dy)

    @staticmethod
    def point_in_rect(point, rect):
        x, y = point
        x1, y1, x2, y2 = rect
        return x1 <= x <= x2 and y1 <= y <= y2

    def stable_confirm(self, key, value, target):
        self.confirm_counter[key] = min(target + 2, self.confirm_counter[key] + 1) if value else max(0, self.confirm_counter[key] - 1)
        return self.confirm_counter[key] >= target

    def _post_alert(self, code, title, severity, frame):
        try:
            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if not ok:
                raise ValueError("JPEG encoding failed")
            payload = {
                "code": code, "title": f"[{code}] {title}", "type": "Accident", "severity": severity,
                "imageUrl": "data:image/jpeg;base64," + base64.b64encode(buf).decode("ascii"),
                "location": CONFIG["camera_name"], "incidentDate": datetime.now(timezone.utc).isoformat(),
                "description": "ثبت خودکار حادثه صنعتی؛ قبل از اقدام اضطراری توسط مسئول ایمنی بررسی شود.",
            }
            response = requests.post(CONFIG["api_url"], json=payload, timeout=CONFIG["api_timeout"], verify=CONFIG["verify_tls"])
            response.raise_for_status()
            log.info("هشدار ارسال شد: %s", code)
        except Exception as exc:
            log.error("ارسال هشدار %s شکست خورد: %s", code, exc)

    def alert(self, code, title, severity, frame):
        if not CONFIG["send_alerts"]:
            return
        with self.alert_lock:
            now = time.monotonic()
            if now - self.last_alert[code] < CONFIG["cooldown_sec"]:
                return
            self.last_alert[code] = now
        self.sender.submit(self._post_alert, code, title, severity, frame.copy())

    def process_people(self, frame, out, frame_id):
        results = self.pose_model.track(frame, persist=True, conf=CONFIG["thresholds"]["pose_conf"], tracker="bytetrack.yaml", verbose=False)
        persons = []
        if results and results[0].boxes is not None and results[0].keypoints is not None:
            boxes = results[0].boxes
            data = results[0].keypoints.data.cpu().numpy()
            for i, kpts in enumerate(data):
                if i >= len(boxes) or kpts.shape[0] < 13:
                    continue
                tid = int(boxes[i].id[0].item()) if boxes[i].id is not None else f"untracked-{i}"
                x1, y1, x2, y2 = map(int, boxes[i].xyxy[0].tolist())
                w, h = max(1, x2 - x1), max(1, y2 - y1)
                shoulder = self.keypoint_center(kpts, 5, 6, CONFIG["thresholds"]["pose_kpt_conf"])
                hip = self.keypoint_center(kpts, 11, 12, CONFIG["thresholds"]["pose_kpt_conf"])
                if shoulder is None or hip is None:
                    continue
                center = ((x1 + x2) / 2, (y1 + y2) / 2)
                current = {"frame": frame_id, "x": center[0], "y": center[1], "angle": self.angle_to_horizontal(shoulder, hip), "ratio": w / h, "time": frame_id / self.fps}
                history = self.person_history[tid]
                history.append(current)
                state = self.person_state[tid]
                state["last_seen"] = current["time"]
                candidate = False
                if len(history) >= 4:
                    old = history[0]
                    dt = current["time"] - old["time"]
                    speed = (current["y"] - old["y"]) / dt if dt > 0 else 0
                    candidate = speed >= CONFIG["thresholds"]["fall_speed_px_sec"] and current["angle"] <= CONFIG["thresholds"]["fall_angle_deg"] and current["ratio"] >= CONFIG["thresholds"]["fall_aspect_ratio"]
                if candidate:
                    state["fall"] += 1
                    state["immobile"] = 0
                else:
                    state["fall"] = max(0, state["fall"] - 1)
                fall = state["fall"] >= CONFIG["thresholds"]["fall_confirm_frames"]
                if fall:
                    cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    cv2.putText(out, f"FALL ID:{tid}", (x1, max(25, y1 - 10)), 0, .65, (0, 0, 255), 2)
                    if not state["alerted"]:
                        state["alerted"] = True
                        self.alert("INC-001", f"سقوط کارگر #{tid}", "High", out)
                elif state["alerted"]:
                    state["immobile"] += 1
                    if len(history) >= int(self.fps * CONFIG["thresholds"]["immobility_sec"]):
                        old = history[0]
                        displacement = math.hypot(center[0] - old["x"], center[1] - old["y"])
                        if displacement <= CONFIG["thresholds"]["immobility_radius_px"] and state["immobile"] >= CONFIG["thresholds"]["immobility_confirm_frames"]:
                            cv2.putText(out, f"IMMOBILE ID:{tid}", (x1, y2 + 20), 0, .6, (0, 165, 255), 2)
                            self.alert("INC-010", f"بی‌حرکتی کارگر #{tid}", "High", out)
                    if state["immobile"] >= CONFIG["thresholds"]["recovery_frames"]:
                        state["alerted"] = False
                        state["immobile"] = 0
                persons.append({"id": tid, "box": (x1, y1, x2, y2), "center": center, "foot": ((x1 + x2) / 2, y2)})
        return persons

    def process_detections(self, frame, out):
        results = self.detect_model(frame, conf=CONFIG["thresholds"]["detect_conf"], verbose=False)
        vehicles, loads = [], []
        if results and results[0].boxes is not None:
            names = results[0].names
            for box in results[0].boxes:
                name = str(names[int(box.cls[0].item())]).lower()
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                item = {"box": (x1, y1, x2, y2), "center": ((x1+x2)/2, (y1+y2)/2), "name": name}
                if name in {"car", "truck", "bus", "train", "forklift"}:
                    vehicles.append(item)
                    cv2.rectangle(out, (x1, y1), (x2, y2), (255, 165, 0), 2)
                elif name in {"backpack", "suitcase", "handbag"}:
                    loads.append(item)
        return vehicles, loads

    def process_fire(self, frame, out):
        results = self.fire_model(frame, conf=CONFIG["thresholds"]["fire_conf"], verbose=False)
        candidates = []
        image_area = frame.shape[0] * frame.shape[1]
        if results and results[0].boxes is not None:
            names = results[0].names
            for box in results[0].boxes:
                name = str(names[int(box.cls[0].item())]).lower()
                if name not in {"fire", "smoke", "flame"}:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                area = max(0, x2-x1) * max(0, y2-y1)
                conf = float(box.conf[0].item())
                if area >= max(CONFIG["thresholds"]["fire_min_area_px"], image_area * CONFIG["thresholds"]["fire_min_area_ratio"]):
                    candidates.append((area, conf, name, (x1, y1, x2, y2)))
        best = max(candidates, default=None)
        detected = best is not None
        # برای جلوگیری از تأیید تشخیص‌های پراکنده، فقط شمارنده متوالی استفاده می‌شود.
        self.fire_state["confirm"] = self.fire_state["confirm"] + 1 if detected else 0
        if detected:
            area, conf, name, (x1, y1, x2, y2) = best
            cv2.rectangle(out, (x1, y1), (x2, y2), (0, 69, 255), 3)
            cv2.putText(out, f"{name.upper()} {conf:.2f}", (x1, max(25, y1-10)), 0, .65, (0, 69, 255), 2)
        confirmed = self.fire_state["confirm"] >= CONFIG["thresholds"]["fire_confirm_frames"]
        if confirmed and not self.fire_state["alerted"]:
            self.fire_state["alerted"] = True
            self.alert("INC-041", "تشخیص پایدار آتش یا دود", "Critical", out)
        elif not detected:
            self.fire_state["recovery"] += 1
            if self.fire_state["recovery"] >= CONFIG["thresholds"]["fire_recovery_frames"]:
                self.fire_state["alerted"] = False
                self.fire_state["recovery"] = 0

    def process_zones_and_collision(self, persons, vehicles, out):
        for person in persons:
            for vehicle in vehicles:
                gap = self.box_gap(person["box"], vehicle["box"])
                if gap <= CONFIG["thresholds"]["collision_dist_px"]:
                    cv2.line(out, tuple(map(int, person["center"])), tuple(map(int, vehicle["center"])), (0, 0, 255), 2)
                    self.alert("INC-012", "خطر نزدیک‌شدن وسیله نقلیه به کارگر", "Critical", out)
        for zone in CONFIG["restricted_zones"]:
            x1, y1, x2, y2 = zone["coords"]
            cv2.rectangle(out, (x1, y1), (x2, y2), (255, 255, 0), 1)
            for person in persons:
                if zone["id"] != "ZONE_EXIT" and self.point_in_rect(person["foot"], zone["coords"]):
                    cv2.putText(out, "DANGER ZONE", (int(person["foot"][0]), int(person["foot"][1])), 0, .55, (0, 0, 255), 2)
                    self.alert("INC-082", f"ورود به {zone['name']}", "High", out)
            if zone["id"] == "ZONE_EXIT" and any(self.point_in_rect(v["center"], zone["coords"]) for v in vehicles):
                self.alert("INC-124", "مسدود شدن مسیر خروج اضطراری", "High", out)

    def process_frame(self, frame, frame_id):
        out = frame.copy()
        persons = self.process_people(frame, out, frame_id)
        vehicles, loads = self.process_detections(frame, out)
        self.process_fire(frame, out)
        self.process_zones_and_collision(persons, vehicles, out)
        cv2.putText(out, f"HSE Steel | frame {frame_id}", (15, 30), 0, .6, (255, 255, 255), 2)
        return out

    def close(self):
        self.sender.shutdown(wait=True)


def main():
    grabber = LatestFrameGrabber(CONFIG["source"])
    cap_probe = cv2.VideoCapture(CONFIG["source"])
    if not cap_probe.isOpened():
        raise RuntimeError(f"ویدیوی نمونه باز نشد: {CONFIG['source']}")
    fps = cap_probe.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_probe.release()
    engine = HSESteelEngine(fps)
    writer = cv2.VideoWriter(CONFIG["output"], cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    frame_id = 0
    try:
        for frame in grabber.frames():
            frame_id += 1
            if frame_id % CONFIG["process_every_n_frames"] != 0:
                continue
            output = engine.process_frame(frame, frame_id)
            writer.write(output)
            if CONFIG["show_gui"]:
                try:
                    cv2.imshow("HSE Steel Fixed", output)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error:
                    CONFIG["show_gui"] = False
            if frame_id % 100 == 0:
                log.info("پردازش فریم %d", frame_id)
    finally:
        grabber.stop()
        writer.release()
        engine.close()
        cv2.destroyAllWindows()
        log.info("خروجی ذخیره شد: %s", CONFIG["output"])


if __name__ == "__main__":
    main()
