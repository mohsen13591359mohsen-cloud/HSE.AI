import os
import math
import time
import base64
import logging
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from threading import Lock

import cv2
import numpy as np
import requests
import torch
from ultralytics import YOLO

# کاهش ریسک کرش/مصرف بالای FFmpeg و OpenCV
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "threads;1")
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


CONFIG = {
    "source": os.getenv("VIDEO_SOURCE", "vid22.mp4"),
    "api_url": os.getenv(
        "SAFETY_API_URL",
        "https://outfit-dimly-juice.ngrok-free.dev/api/SafetyIncidents/camera",
    ),
    "verify_tls": os.getenv("VERIFY_TLS", "true").lower() == "true",
    "cooldown_sec": 15.0,
    "request_timeout_sec": 4.0,
    "person_history_size": 20,
    "track_expire_sec": 5.0,
    "thresholds": {
        # این مقدار بر حسب پیکسل در ثانیه و با FPS ویدیو محاسبه می‌شود.
        "fall_speed_px_sec": 160.0,
        "spine_angle_horizon": 35.0,
        "aspect_ratio_fall": 1.10,
        "fall_confirm_frames": 3,
        "fall_recovery_frames": 8,
        "pose_keypoint_conf": 0.45,
        "fire_conf": 0.45,
        "fire_confirm_frames": 3,
        "fire_recovery_frames": 8,
        "fire_history_size": 10,
        # آتش کوچک در صورت تداوم نیز معتبر است؛ رشد شرط الزامی نیست.
        "fire_min_area_px": 300,
        "fire_min_area_ratio": 0.001,
        "fire_growth_rate": 1.35,
    },
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(message)s",
)
log = logging.getLogger("HSE_Engine")


class HighPrecisionAccidentEngine:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("اجرای موتور روی: %s", self.device.upper())

        self.pose_model = YOLO("yolov8m-pose.pt").to(self.device)
        self.fire_model = self._load_fire_model()

        # تاریخچه هر فرد جداگانه است.
        self.person_history = defaultdict(
            lambda: deque(maxlen=CONFIG["person_history_size"])
        )
        self.person_state = defaultdict(
            lambda: {
                "candidate_frames": 0,
                "recovery_frames": 0,
                "alerted": False,
                "last_seen": 0.0,
            }
        )

        # تاریخچه فقط مربوط به بزرگ‌ترین ناحیه معتبر آتش در هر فریم است.
        self.fire_history = deque(
            maxlen=CONFIG["thresholds"]["fire_history_size"]
        )
        self.fire_candidate_frames = 0
        self.fire_recovery_frames = 0
        self.fire_alerted = False

        self.last_alert_time = defaultdict(float)
        self.alert_lock = Lock()
        self.sender = ThreadPoolExecutor(max_workers=2)

    @staticmethod
    def _load_fire_model():
        model_path = os.getenv("FIRE_MODEL_PATH", "fire_smoke_yolov8s.pt")

        if not os.path.exists(model_path):
            model_url = os.getenv(
                "FIRE_MODEL_URL",
                "https://huggingface.co/fatihakturk/yolov8-fire-and-smoke-detection/resolve/main/best.pt",
            )
            log.warning("مدل آتش پیدا نشد؛ دانلود از سرور آغاز می‌شود.")
            try:
                with requests.get(
                    model_url,
                    stream=True,
                    timeout=(5, 120),
                ) as response:
                    response.raise_for_status()
                    with open(model_path, "wb") as output:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                output.write(chunk)
                log.info("مدل آتش با موفقیت دانلود شد: %s", model_path)
            except Exception as exc:
                raise RuntimeError(
                    "مدل آتش قابل دریافت نیست؛ تشخیص آتش غیرفعال نشد و برنامه متوقف شد."
                ) from exc

        try:
            model = YOLO(model_path)
            log.info("مدل Fire/Smoke بارگذاری شد.")
            return model
        except Exception as exc:
            raise RuntimeError(f"بارگذاری مدل آتش شکست خورد: {exc}") from exc

    @staticmethod
    def calculate_spine_angle(shoulder, hip):
        dx = float(hip[0] - shoulder[0])
        dy = float(hip[1] - shoulder[1])
        # صفر درجه یعنی بدن افقی و ۹۰ درجه یعنی بدن عمودی.
        return math.degrees(math.atan2(abs(dy), abs(dx) + 1e-6))

    @staticmethod
    def _mean_keypoints(kpts, first, second, min_conf):
        """مرکز دو keypoint را فقط در صورت معتبر بودن confidence برمی‌گرداند."""
        if kpts.shape[0] < 3:
            return None
        if kpts[first][2] < min_conf or kpts[second][2] < min_conf:
            return None
        return (
            (float(kpts[first][0]) + float(kpts[second][0])) / 2.0,
            (float(kpts[first][1]) + float(kpts[second][1])) / 2.0,
        )

    def _person_fall_candidate(self, history, current, fps):
        if len(history) < 4:
            return False

        # استفاده از بازه فریمی ثابت، نه مدت زمان کند/سریع بودن inference.
        old = history[0]
        frame_delta = current["frame"] - old["frame"]
        if frame_delta <= 0 or fps <= 0:
            return False

        elapsed = frame_delta / fps
        vertical_speed = (current["center_y"] - old["center_y"]) / elapsed
        thresholds = CONFIG["thresholds"]

        # سقوط معمولاً با حرکت رو به پایین، افقی‌شدن تنه و افزایش W/H همراه است.
        return (
            vertical_speed >= thresholds["fall_speed_px_sec"]
            and current["spine_angle"] <= thresholds["spine_angle_horizon"]
            and current["aspect_ratio"] >= thresholds["aspect_ratio_fall"]
        )

    def _send_alert_request(self, code, title, severity, frame):
        try:
            ok, buf = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70]
            )
            if not ok:
                raise ValueError("تبدیل فریم به JPEG شکست خورد")

            payload = {
                "code": code,
                "title": f"🚨 {title}",
                "severity": severity,
                "imageUrl": "data:image/jpeg;base64,"
                + base64.b64encode(buf).decode("ascii"),
                "incidentDate": datetime.now(timezone.utc).isoformat(),
                "description": f"تشخیص هوشمند حادثه | کد: {code}",
            }

            response = requests.post(
                CONFIG["api_url"],
                json=payload,
                timeout=CONFIG["request_timeout_sec"],
                verify=CONFIG["verify_tls"],
            )
            response.raise_for_status()
            log.warning("هشدار با موفقیت ارسال شد: %s", code)
        except Exception as exc:
            log.error("ارسال هشدار %s شکست خورد: %s", code, exc)

    def send_alert(self, code, title, severity, frame):
        """ارسال محدودشده و غیرهمگام؛ cooldown فقط هنگام رزرو هشدار اعمال می‌شود."""
        now = time.monotonic()
        with self.alert_lock:
            if now - self.last_alert_time[code] < CONFIG["cooldown_sec"]:
                return False
            self.last_alert_time[code] = now

        self.sender.submit(
            self._send_alert_request,
            code,
            title,
            severity,
            frame.copy(),
        )
        return True

    def _cleanup_old_tracks(self, now):
        expiry = CONFIG["track_expire_sec"]
        old_ids = [
            tid
            for tid, state in self.person_state.items()
            if now - state["last_seen"] > expiry
        ]
        for tid in old_ids:
            self.person_state.pop(tid, None)
            self.person_history.pop(tid, None)

    def _process_people(self, frame, annotated, frame_index, fps, now):
        results = self.pose_model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )
        if not results or results[0].keypoints is None or results[0].boxes is None:
            self._cleanup_old_tracks(now)
            return annotated

        boxes = results[0].boxes
        keypoints = results[0].keypoints.data.cpu().numpy()
        min_kpt_conf = CONFIG["thresholds"]["pose_keypoint_conf"]

        for i, kpts in enumerate(keypoints):
            if kpts.shape[0] < 13 or i >= len(boxes):
                continue

            track_id = (
                int(boxes[i].id[0].item())
                if boxes[i].id is not None
                else f"untracked-{i}"
            )
            x1, y1, x2, y2 = map(int, boxes[i].xyxy[0].tolist())
            width = max(1, x2 - x1)
            height = max(1, y2 - y1)

            shoulder = self._mean_keypoints(kpts, 5, 6, min_kpt_conf)
            hip = self._mean_keypoints(kpts, 11, 12, min_kpt_conf)
            if shoulder is None or hip is None:
                continue

            current = {
                "center_y": (y1 + y2) / 2.0,
                "spine_angle": self.calculate_spine_angle(shoulder, hip),
                "aspect_ratio": width / float(height),
                "frame": frame_index,
            }
            history = self.person_history[track_id]
            history.append(current)

            state = self.person_state[track_id]
            state["last_seen"] = now
            candidate = self._person_fall_candidate(history, current, fps)

            if candidate:
                state["candidate_frames"] += 1
                state["recovery_frames"] = 0
            else:
                state["candidate_frames"] = max(0, state["candidate_frames"] - 1)
                if state["alerted"]:
                    state["recovery_frames"] += 1
                    if state["recovery_frames"] >= CONFIG["thresholds"]["fall_recovery_frames"]:
                        state["alerted"] = False
                        state["recovery_frames"] = 0

            confirmed = (
                state["candidate_frames"]
                >= CONFIG["thresholds"]["fall_confirm_frames"]
            )
            if confirmed and not state["alerted"]:
                state["alerted"] = True
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(
                    annotated,
                    f"FALL DETECTED ID:{track_id}",
                    (x1, max(25, y1 - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
                self.send_alert(
                    "ACC-FALL",
                    f"سقوط شدید کارگر کد #{track_id}",
                    "Critical",
                    annotated,
                )

        self._cleanup_old_tracks(now)
        return annotated

    def _process_fire(self, frame, annotated, now):
        results = self.fire_model(
            frame,
            conf=CONFIG["thresholds"]["fire_conf"],
            verbose=False,
        )
        candidates = []
        image_area = frame.shape[0] * frame.shape[1]

        if results and results[0].boxes is not None:
            names = results[0].names
            for box in results[0].boxes:
                cls_id = int(box.cls[0].item())
                class_name = str(names[cls_id]).lower()
                if class_name not in {"fire", "smoke", "flame"}:
                    continue

                fx1, fy1, fx2, fy2 = map(int, box.xyxy[0].tolist())
                area = max(0, fx2 - fx1) * max(0, fy2 - fy1)
                confidence = float(box.conf[0].item())
                min_area = max(
                    CONFIG["thresholds"]["fire_min_area_px"],
                    image_area * CONFIG["thresholds"]["fire_min_area_ratio"],
                )
                if area < min_area:
                    continue

                candidates.append(
                    {
                        "area": area,
                        "confidence": confidence,
                        "name": class_name,
                        "box": (fx1, fy1, fx2, fy2),
                    }
                )

        # هر فریم فقط بزرگ‌ترین ناحیه را وارد تاریخچه می‌کنیم؛ اشیای مختلف قاطی نمی‌شوند.
        largest = max(candidates, key=lambda item: item["area"], default=None)
        fire_in_frame = largest is not None
        self.fire_history.append((largest["area"] if largest else 0, now))

        if fire_in_frame:
            fx1, fy1, fx2, fy2 = largest["box"]
            cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), (0, 69, 255), 3)
            cv2.putText(
                annotated,
                f"{largest['name'].upper()} {largest['confidence']:.2f}",
                (fx1, max(25, fy1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 69, 255),
                2,
            )
            self.fire_candidate_frames += 1
            self.fire_recovery_frames = 0
        else:
            self.fire_candidate_frames = max(0, self.fire_candidate_frames - 1)
            if self.fire_alerted:
                self.fire_recovery_frames += 1
                if self.fire_recovery_frames >= CONFIG["thresholds"]["fire_recovery_frames"]:
                    self.fire_alerted = False
                    self.fire_recovery_frames = 0

        confirmed = (
            self.fire_candidate_frames
            >= CONFIG["thresholds"]["fire_confirm_frames"]
        )
        if confirmed and not self.fire_alerted:
            self.fire_alerted = True
            self.send_alert(
                "ACC-FIRE",
                "کشف آتش‌سوزی یا دود",
                "Critical",
                annotated,
            )
        return annotated

    def process_frame(self, frame, frame_index, fps):
        now = time.monotonic()
        annotated = frame.copy()
        annotated = self._process_people(
            frame, annotated, frame_index, fps, now
        )
        annotated = self._process_fire(frame, annotated, now)
        cv2.putText(
            annotated,
            f"HSE Engine | frame={frame_index} | FPS={fps:.1f}",
            (15, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
        return annotated

    def shutdown(self):
        self.sender.shutdown(wait=True)


def main():
    cap = cv2.VideoCapture(CONFIG["source"])
    if not cap.isOpened():
        raise RuntimeError(f"باز کردن منبع ویدیو شکست خورد: {CONFIG['source']}")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = source_fps if source_fps and source_fps > 1 else 25.0
    engine = HighPrecisionAccidentEngine()
    is_headless = os.environ.get("DISPLAY") is None
    frame_index = 0

    try:
        log.info("سیستم آماده است؛ FPS مبنا: %.2f", fps)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_index += 1
            processed = engine.process_frame(frame, frame_index, fps)

            if not is_headless:
                cv2.imshow("HSE Fire and Fall Detector", processed)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            elif frame_index % 100 == 0:
                log.info("فریم پردازش‌شده: %d", frame_index)
    finally:
        cap.release()
        engine.shutdown()
        if not is_headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
