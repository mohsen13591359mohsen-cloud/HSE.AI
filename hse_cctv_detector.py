"""
HSE CCTV Detector — نسخه حرفه‌ای و کاربردی
مشکلات نسخه قبل که حل شد:
  ✅ Cooldown واقعی — هر حادثه مستقل cooldown دارد
  ✅ حذف duplicate ثبت (0s, 11s, 12s بیهوشی)
  ✅ بهبود تشخیص سقوط با تاریخچه واقعی
  ✅ حریم خطرناک قابل تنظیم از خارج
  ✅ لاگ محلی برای debug
  ✅ Skip فریم برای سرعت بیشتر
  ✅ Auto-reconnect برای ویدیو/RTSP
"""

import cv2
import numpy as np
from ultralytics import YOLO
import requests
import time
import math
import base64
from datetime import datetime
import urllib3
from collections import defaultdict, deque
import logging

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ═══════════════════════════════════════════════════════════
# ⚙️ تنظیمات اصلی
# ═══════════════════════════════════════════════════════════
BASE_NGROK_URL = "https://outfit-dimly-juice.ngrok-free.dev"
API_URL        = f"{BASE_NGROK_URL}/api/SafetyIncidents/camera"
CAMERA_NAME    = "دوربین ۱ - سوله اصلی HSE"
VIDEO_SOURCE   = "vid22.mp4"   # یا آدرس RTSP: "rtsp://user:pass@192.168.1.100:554/stream"

# ── تنظیمات Cooldown مستقل برای هر حادثه ───────────────
# هر کلید = یک نوع حادثه — مستقل از بقیه
COOLDOWN = {
    "fall":       60,   # سقوط — حداقل ۶۰ ثانیه بین دو ثبت
    "fire":       45,   # آتش
    "zone":       30,   # ورود به حریم
    "immobility": 90,   # بیهوشی — باید مطمئن شویم
    "collision":  45,   # برخورد
}

# ── تنظیمات تشخیص ───────────────────────────────────────
CONF_THRESHOLD   = 0.50   # ← بالاتر از قبل برای کاهش false positive
FALL_RATIO       = 1.6    # نسبت عرض/ارتفاع برای سقوط
FALL_SPEED       = 100    # پیکسل/ثانیه — بالاتر = کمتر false positive
IMMOBILITY_SEC   = 12.0   # ← افزایش از 8 به 12 ثانیه
IMMOBILITY_PIXEL = 20
FIRE_MIN_PIXELS  = 6000
COLLISION_DIST   = 90
CONFIRM_FRAMES   = 4      # ← افزایش از 3 به 4 فریم
FALL_CONFIRM     = 3
SKIP_FRAMES      = 2      # پردازش هر N فریم — برای سرعت

# ── لاگ ─────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('hse_detector.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("HSE")

# ═══════════════════════════════════════════════════════════
print("=" * 70)
print("🚀 HSE Detector — نسخه حرفه‌ای با Cooldown مستقل")
print("=" * 70)

model = YOLO("yolov8s.pt")
cap   = cv2.VideoCapture(VIDEO_SOURCE)
fps   = cap.get(cv2.CAP_PROP_FPS) or 25

log.info(f"ویدیو: {VIDEO_SOURCE} | FPS: {fps:.0f} | Skip: هر {SKIP_FRAMES} فریم")

# ─── State ───────────────────────────────────────────────
# Cooldown مستقل — هر incident آخرین زمان ارسال خود را دارد
last_sent       = defaultdict(float)   # {incident_key: timestamp}
active_titles   = {}                   # {incident_key: title} — برای لاگ

# ردیابی
person_history  = defaultdict(lambda: deque(maxlen=40))
confirm_counter = defaultdict(int)
prev_fire_mask  = None
frame_idx       = 0

# ─── حریم‌های خطرناک ─────────────────────────────────────
# می‌توانید چند ناحیه تعریف کنید
def build_zones(h, w):
    return [
        # ناحیه اصلی سمت چپ
        np.array([
            [50,           int(h * 0.30)],
            [int(w * 0.42), int(h * 0.30)],
            [int(w * 0.42), int(h * 0.92)],
            [50,           int(h * 0.92)]
        ], np.int32),
    ]

# ─── توابع کمکی ──────────────────────────────────────────
def frame_to_b64(frame):
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 65])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()

def in_any_zone(cx, cy, zones):
    return any(
        cv2.pointPolygonTest(z, (float(cx), float(cy)), False) >= 0
        for z in zones
    )

def can_send(key):
    """آیا می‌توان این حادثه را ارسال کرد؟"""
    return time.time() - last_sent[key] >= COOLDOWN.get(key, 30)

def update_confirm(key, detected):
    if detected:
        confirm_counter[key] = min(confirm_counter[key] + 1, CONFIRM_FRAMES + 2)
    else:
        # کاهش تدریجی‌تر — یک فریم منفی همه چیز را خراب نکند
        confirm_counter[key] = max(confirm_counter[key] - 1, 0)
    thr = FALL_CONFIRM if key == "fall" else CONFIRM_FRAMES
    return confirm_counter[key] >= thr

def detect_fire(frame):
    """تشخیص آتش با YCrCb + HSV + Flicker"""
    global prev_fire_mask

    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)
    mask_ycrcb = ((Y > 170) & (Cr > 145) & (Cb < 115) & (Cr > Cb)).astype(np.uint8) * 255

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m1  = cv2.inRange(hsv, np.array([0,  140, 180], np.uint8), np.array([25, 255, 255], np.uint8))
    m2  = cv2.inRange(hsv, np.array([160,140, 180], np.uint8), np.array([179,255, 255], np.uint8))
    mask_hsv = cv2.bitwise_or(m1, m2)

    combined = cv2.bitwise_and(mask_ycrcb, mask_hsv)
    kernel   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
    pixels   = cv2.countNonZero(combined)

    is_flickering = False
    if prev_fire_mask is not None and pixels > FIRE_MIN_PIXELS:
        diff        = cv2.absdiff(combined, prev_fire_mask)
        flicker_r   = cv2.countNonZero(diff) / float(pixels)
        is_flickering = 0.08 < flicker_r < 0.80

    prev_fire_mask = combined.copy()
    return (pixels > FIRE_MIN_PIXELS and is_flickering), pixels

def check_fall(pid, cx, cy, ratio, now):
    hist = person_history[pid]
    if len(hist) < 5: return False

    ratio_ok = ratio > FALL_RATIO

    # بررسی سرعت نزولی با پنجره بزرگتر
    old = hist[-5]
    dt  = now - old[2]
    vy  = (cy - old[1]) / dt if dt > 0.001 else 0
    speed_ok = vy > FALL_SPEED

    # بررسی تغییر ناگهانی در ratio (از عمودی به افقی)
    old_ratio = old[3]
    ratio_change = ratio - old_ratio > 0.4  # تغییر سریع به افقی

    return ratio_ok and (speed_ok or ratio_change)

def check_immobility(pid, now):
    hist = person_history[pid]
    if len(hist) < 15: return False, 0

    # بررسی در کل تاریخچه — نه فقط آخرین
    oldest = hist[0]
    cx, cy = hist[-1][0], hist[-1][1]
    dist   = math.hypot(cx - oldest[0], cy - oldest[1])
    dur    = now - oldest[2]

    # بررسی میانه هم — اگر در بین راه حرکت کرده باشد، بیهوش نیست
    mid = hist[len(hist)//2]
    mid_dist = math.hypot(cx - mid[0], cy - mid[1])

    truly_still = dist < IMMOBILITY_PIXEL and mid_dist < IMMOBILITY_PIXEL
    return (truly_still and dur > IMMOBILITY_SEC), dur

def send_incident(key, title, inc_type, severity, frame):
    """ارسال حادثه با بررسی Cooldown مستقل"""
    if not can_send(key):
        remaining = COOLDOWN.get(key, 30) - (time.time() - last_sent[key])
        log.debug(f"⏳ {key} در cooldown — {remaining:.0f}s مانده")
        return False

    try:
        payload = {
            "title":       title,
            "type":        inc_type,
            "severity":    severity,
            "imageUrl":    frame_to_b64(frame),
            "location":    CAMERA_NAME,
            "department":  "ایمنی و HSE",
            "incidentDate":datetime.now().isoformat(),
            "description": f"تشخیص هوشمند AI ({CONFIRM_FRAMES} فریم تأیید) | {title}"
        }
        r = requests.post(API_URL, json=payload, verify=False, timeout=8)
        if r.status_code in (200, 201):
            last_sent[key] = time.time()
            active_titles[key] = title
            log.info(f"✅ ثبت شد: [{key}] {title}")
            return True
        else:
            log.warning(f"⚠️ API {r.status_code}: {key}")
            return False
    except Exception as e:
        log.error(f"❌ خطا [{key}]: {e}")
        return False

# ═══════════════════════════════════════════════════════════
# 🔁 حلقه اصلی
# ═══════════════════════════════════════════════════════════
reconnect_count = 0

while True:
    ok, frame = cap.read()

    # ── Auto-reconnect ──────────────────────────────────
    if not ok:
        reconnect_count += 1
        if reconnect_count > 3:
            log.warning("📹 ویدیو تمام شد — restart")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            person_history.clear()
            confirm_counter.clear()
            prev_fire_mask = None
            reconnect_count = 0
        time.sleep(0.1)
        continue

    reconnect_count = 0
    frame_idx += 1

    # ── Skip frames برای سرعت ──────────────────────────
    if frame_idx % SKIP_FRAMES != 0:
        continue

    h, w = frame.shape[:2]
    now  = time.time()

    # ── رسم حریم‌های خطرناک ────────────────────────────
    zones = build_zones(h, w)
    for zone in zones:
        overlay = frame.copy()
        cv2.fillPoly(overlay, [zone], (0, 0, 180))
        cv2.addWeighted(overlay, 0.18, frame, 0.82, 0, frame)
        cv2.polylines(frame, [zone], True, (0, 0, 255), 2)

    # ── YOLO + ByteTrack ───────────────────────────────
    try:
        results = model.track(
            frame, persist=True,
            conf=CONF_THRESHOLD, iou=0.45,
            tracker="bytetrack.yaml", verbose=False
        )
    except Exception as e:
        log.error(f"YOLO error: {e}")
        continue

    persons  = []
    vehicles = []

    for r in results:
        if r.boxes is None: continue
        for box in r.boxes:
            cls  = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            if conf < CONF_THRESHOLD: continue

            x1,y1,x2,y2 = map(int, box.xyxy[0])
            cx, cy = (x1+x2)//2, (y1+y2)//2
            bw, bh = x2-x1, y2-y1
            ratio  = bw / float(bh) if bh > 0 else 0
            tid    = int(box.id[0]) if box.id is not None else id(box)

            if cls == "person":
                persons.append({'id':tid,'box':(x1,y1,x2,y2),'center':(cx,cy),'ratio':ratio})
                person_history[tid].append((cx, cy, now, ratio))
                color = (0,255,0)
            elif cls in ("car","truck","bus","motorcycle","forklift"):
                vehicles.append({'box':(x1,y1,x2,y2),'center':(cx,cy)})
                color = (255,165,0)
            else:
                continue

            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 1)
            cv2.putText(frame, f"{cls} {conf:.0%} #{tid}",
                       (x1, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)

    # ══ تشخیص حوادث — هر قانون مستقل است ══════════════

    # ── قانون ۱: سقوط ───────────────────────────────────
    fall_now = any(
        check_fall(p['id'], p['center'][0], p['center'][1], p['ratio'], now)
        for p in persons
    )
    if update_confirm("fall", fall_now):
        # پیدا کردن کارگری که افتاده
        fallen = next((p for p in persons if check_fall(p['id'],p['center'][0],p['center'][1],p['ratio'],now)), None)
        if fallen:
            fx1,fy1,fx2,fy2 = fallen['box']
            cv2.rectangle(frame,(fx1,fy1),(fx2,fy2),(0,0,255),3)
        send_incident("fall", "حادثه: سقوط و زمین‌خوردن کارگر", "Accident", "High", frame)

    # ── قانون ۲: آتش ─────────────────────────────────────
    fire_ok, fire_px = detect_fire(frame)
    if update_confirm("fire", fire_ok):
        send_incident("fire", f"حادثه: تشخیص آتش/حریق", "Accident", "High", frame)

    # ── قانون ۳: ورود به حریم ────────────────────────────
    # فقط اگر شخص واقعاً داخل باشد — نه فقط لبه
    zone_persons = [p for p in persons if in_any_zone(*p['center'], zones)]
    if update_confirm("zone", len(zone_persons) > 0):
        send_incident("zone", f"خطر: ورود غیرمجاز به حریم — {len(zone_persons)} نفر",
                      "Hazard", "Medium", frame)

    # ── قانون ۴: بی‌حرکتی ────────────────────────────────
    # مستقل از سایر قانون‌ها — cooldown 90 ثانیه
    for p in persons:
        is_still, dur = check_immobility(p['id'], now)
        if update_confirm(f"immobility_{p['id']}", is_still):
            if can_send("immobility"):
                # هایلایت کارگر بی‌حرکت
                x1,y1,x2,y2 = p['box']
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,165,255),3)
                cv2.putText(frame,"STILL!", (x1,y1-8),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6,(0,165,255),2)
                send_incident("immobility",
                    f"حادثه: بی‌حرکتی {dur:.0f} ثانیه — احتمال بیهوشی",
                    "Accident", "High", frame)
            break  # فقط یک بار در هر فریم

    # ── قانون ۵: برخورد خودرو-کارگر ─────────────────────
    col_now = False
    for p in persons:
        px,py = p['center']
        for v in vehicles:
            if math.hypot(px-v['center'][0], py-v['center'][1]) < COLLISION_DIST:
                col_now = True
                break
    if update_confirm("collision", col_now):
        send_incident("collision", "خطر: احتمال برخورد لیفتراک با کارگر",
                      "NearMiss", "High", frame)

    # ── نمایش وضعیت ────────────────────────────────────
    ts    = datetime.now().strftime("%H:%M:%S")
    color = (0,0,255) if any(confirm_counter[k]>0 for k in ["fall","fire","zone","immobility","collision"]) \
            else (0,255,200)
    cv2.putText(frame,
        f"HSE AI | {len(persons)}p {len(vehicles)}v | {ts}",
        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    # وضعیت cooldown
    for i, (key, cd) in enumerate(COOLDOWN.items()):
        remaining = max(0, cd - (now - last_sent[key]))
        status    = f"{key}: {'✅ ready' if remaining==0 else f'⏳{remaining:.0f}s'}"
        cv2.putText(frame, status, (10, 45 + i*16),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                   (0,255,0) if remaining==0 else (128,128,128), 1)

    time.sleep(0.02)

cap.release()
log.info("✅ پردازش تمام شد")
