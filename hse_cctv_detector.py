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

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ═══════════════════════════════════════════════════════════
# ⚙️ تنظیمات
# ═══════════════════════════════════════════════════════════
BASE_NGROK_URL   = "https://outfit-dimly-juice.ngrok-free.dev"
API_URL        = f"{BASE_NGROK_URL}/api/SafetyIncidents/camera"
CAMERA_NAME      = "دوربین ۱ - سوله اصلی HSE"
VIDEO_SOURCE     = "vid22.mp4"
COOLDOWN_SECONDS = 20

# ── تنظیمات دقت ─────────────────────────────────────────
CONF_THRESHOLD   = 0.45   # حداقل اطمینان YOLO
FALL_RATIO       = 1.5    # نسبت عرض/ارتفاع برای سقوط
FALL_SPEED       = 80     # پیکسل/ثانیه حرکت به پایین
IMMOBILITY_SEC   = 8.0    # ثانیه بی‌حرکتی (احتمال بیهوشی)
IMMOBILITY_PIXEL = 15     # پیکسل تحمل برای بی‌حرکتی
FIRE_MIN_PIXELS  = 5000   # حداقل پیکسل آتش
COLLISION_DIST   = 100    # فاصله برخورد خودرو-کارگر
CONFIRM_FRAMES   = 3      # تعداد فریم متوالی برای تأیید حادثه
FALL_CONFIRM     = 2      # فریم تأیید سقوط

# ═══════════════════════════════════════════════════════════
# 🧠 بارگذاری مدل
# ═══════════════════════════════════════════════════════════
print("=" * 70)
print("🚀 HSE Detector — نسخه بهینه با ردیابی واقعی و تشخیص پیشرفته آتش")
print("=" * 70)

model = YOLO("yolov8s.pt")

cap = cv2.VideoCapture(VIDEO_SOURCE)
fps = cap.get(cv2.CAP_PROP_FPS) or 25

print(f"📹 ویدیو: {VIDEO_SOURCE} | FPS: {fps:.0f}")
print(f"🤖 مدل: yolov8s | Conf: {CONF_THRESHOLD}")

# ─── State ───────────────────────────────────────────────
last_alert_time       = 0
active_incident_title = None
prev_fire_mask        = None  # برای بررسی نوسان آتش

# ردیابی با ID واقعی (ByteTrack)
person_history = defaultdict(lambda: deque(maxlen=30))

# تأیید چند فریمی
confirm_counter = defaultdict(int)

# ─── توابع کمکی ──────────────────────────────────────────
def frame_to_b64(frame):
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()

def in_zone(cx, cy, poly):
    return cv2.pointPolygonTest(poly, (float(cx), float(cy)), False) >= 0

def update_confirm(key, detected):
    """تأیید چند فریمی — جلوگیری از false positive"""
    if detected:
        confirm_counter[key] = min(confirm_counter[key] + 1, CONFIRM_FRAMES + 1)
    else:
        confirm_counter[key] = max(confirm_counter[key] - 1, 0)
    threshold = FALL_CONFIRM if key == "fall" else CONFIRM_FRAMES
    return confirm_counter[key] >= threshold

def detect_fire(frame):
    """
    تشخیص هوشمند آتش با ترکیب YCrCb + HSV + آنالیز درخشندگی و نوسان (Flicker)
    حذف کامل خطای تشخیص روی چوب، لباس کارگر و چراغ‌ها
    """
    global prev_fire_mask

    # ۱. بررسی درخشندگی و شدت رنگ در فضای YCrCb
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    Y, Cr, Cb = cv2.split(ycrcb)

    # آتش واقعی درخشندگی بالا (Y>170) و تفاضل شدید Cr و Cb دارد
    fire_mask_ycrcb = (Y > 170) & (Cr > 145) & (Cb < 115) & (Cr > Cb)
    mask_ycrcb = (fire_mask_ycrcb * 255).astype(np.uint8)

    # ۲. فیلتر مکمل HSV با Saturation و Value بالا (حذف اجسام کدر)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array([0, 140, 180], np.uint8), np.array([25, 255, 255], np.uint8))
    m2 = cv2.inRange(hsv, np.array([160, 140, 180], np.uint8), np.array([179, 255, 255], np.uint8))
    mask_hsv = cv2.bitwise_or(m1, m2)

    # ترکیب هر دو فیلتر
    combined_mask = cv2.bitwise_and(mask_ycrcb, mask_hsv)

    # مورفولوژی برای حذف نویز
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)

    current_pixels = cv2.countNonZero(combined_mask)

    # ۳. آنالیز نوسان و پویایی (Flicker) — اجسام ثابت مثل جعبه یا لباس نوسان ندارند
    is_flickering = False
    if prev_fire_mask is not None and current_pixels > FIRE_MIN_PIXELS:
        diff = cv2.absdiff(combined_mask, prev_fire_mask)
        diff_pixels = cv2.countNonZero(diff)
        
        # میزان تغییر لبه‌های شعله نسبت به کل مساحت
        flicker_ratio = diff_pixels / float(current_pixels)
        if 0.10 < flicker_ratio < 0.75:
            is_flickering = True

    prev_fire_mask = combined_mask.copy()

    # شرط نهایی: مساحت کافی + نوسان واقعی شعله
    is_fire = (current_pixels > FIRE_MIN_PIXELS) and is_flickering
    return is_fire, current_pixels

def check_fall(pid, cx, cy, ratio, now):
    """تشخیص سقوط با دو شرط نسبت ابعاد + سرعت نزولی"""
    hist = person_history[pid]
    if len(hist) < 3:
        return False

    ratio_ok = ratio > FALL_RATIO
    old_cy, old_t = hist[-3][1], hist[-3][2]
    dt = now - old_t
    vy = (cy - old_cy) / dt if dt > 0.001 else 0
    speed_ok = vy > FALL_SPEED

    return ratio_ok and speed_ok

def check_immobility(pid, now):
    """تشخیص بی‌حرکتی طولانی مدت"""
    hist = person_history[pid]
    if len(hist) < 10:
        return False, 0

    oldest = hist[0]
    ox, oy, ot = oldest[0], oldest[1], oldest[2]
    cx, cy = hist[-1][0], hist[-1][1]
    dist = math.hypot(cx - ox, cy - oy)
    dur  = now - ot

    return (dist < IMMOBILITY_PIXEL and dur > IMMOBILITY_SEC), dur

# ═══════════════════════════════════════════════════════════
# 🔁 حلقه اصلی
# ═══════════════════════════════════════════════════════════
frame_idx = 0

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        person_history.clear()
        confirm_counter.clear()
        active_incident_title = None
        prev_fire_mask = None
        continue

    h, w = frame.shape[:2]
    now  = time.time()
    frame_idx += 1

    # ── حریم خطرناک ────────────────────────────────────
    danger_zone = np.array([
        [50,        int(h * 0.3)],
        [int(w * 0.4), int(h * 0.3)],
        [int(w * 0.4), int(h * 0.9)],
        [50,        int(h * 0.9)]
    ], np.int32)

    overlay = frame.copy()
    cv2.fillPoly(overlay, [danger_zone], (0, 0, 180))
    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
    cv2.polylines(frame, [danger_zone], True, (0, 0, 255), 2)

    # ── YOLO با ردیابی ByteTrack ──────────────────────
    results = model.track(
        frame,
        persist   = True,
        conf      = CONF_THRESHOLD,
        iou       = 0.5,
        tracker   = "bytetrack.yaml",
        verbose   = False
    )

    persons  = []
    vehicles = []

    for r in results:
        if r.boxes is None: continue
        for box in r.boxes:
            cls  = model.names[int(box.cls[0])]
            conf = float(box.conf[0])
            if conf < CONF_THRESHOLD: continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1+x2)//2, (y1+y2)//2
            bw, bh = x2-x1, y2-y1
            ratio  = bw / float(bh) if bh > 0 else 0

            tid = int(box.id[0]) if box.id is not None else id(box)

            if cls == "person":
                persons.append({
                    'id': tid, 'box': (x1,y1,x2,y2),
                    'center': (cx,cy), 'w': bw, 'h': bh, 'ratio': ratio
                })
                person_history[tid].append((cx, cy, now, ratio))

            elif cls in ("car","truck","bus","motorcycle"):
                vehicles.append({'box':(x1,y1,x2,y2), 'center':(cx,cy)})

            color = (0,255,0) if cls=="person" else (255,165,0)
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 1)
            cv2.putText(frame, f"{cls} {conf:.0%}",
                       (x1, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # ── تشخیص حوادث ─────────────────────────────────
    incident = None

    # ── قانون ۱: سقوط ──
    fall_now = False
    for p in persons:
        if check_fall(p['id'], p['center'][0], p['center'][1], p['ratio'], now):
            fall_now = True
            break
    if update_confirm("fall", fall_now):
        incident = ("حادثه: سقوط و زمین‌خوردن کارگر", "Accident", "High", "fall")

    # ── قانون ۲: آتش ─────────
    if not incident:
        fire_ok, fire_px = detect_fire(frame)
        if update_confirm("fire", fire_ok):
            incident = (f"حادثه: تشخیص آتش ({fire_px//100}۰۰ px)", "Accident", "High", "fire")

    # ── قانون ۳: ورود به حریم ────────────────────────
    if not incident:
        zone_now = any(in_zone(*p['center'], danger_zone) for p in persons)
        if update_confirm("zone", zone_now):
            incident = ("خطر: ورود غیرمجاز به حریم خطرناک", "Hazard", "Medium", "zone")

    # ── قانون ۴: بی‌حرکتی ─────────────────────
    if not incident:
        imm_now = False
        imm_dur = 0
        for p in persons:
            ok_imm, dur = check_immobility(p['id'], now)
            if ok_imm:
                imm_now = True
                imm_dur = dur
                break
        if update_confirm("immobility", imm_now):
            incident = (
                f"حادثه: بی‌حرکتی {imm_dur:.0f}s — احتمال بیهوشی",
                "Accident", "High", "immobility"
            )

    # ── قانون ۵: برخورد ─────────────────
    if not incident:
        col_now = False
        for p in persons:
            px,py = p['center']
            for v in vehicles:
                vx,vy = v['center']
                if math.hypot(px-vx, py-vy) < COLLISION_DIST:
                    col_now = True
                    break
        if update_confirm("collision", col_now):
            incident = ("خطر: احتمال برخورد لیفتراک با کارگر", "NearMiss", "High", "collision")

    # ── ارسال به API ──────────────────────────────────
    if incident:
        title, inc_type, severity, key = incident
        print(f"\a🚨 [{datetime.now():%H:%M:%S}] {title}")

        if (title != active_incident_title and
                now - last_alert_time > COOLDOWN_SECONDS):
            try:
                payload = {
                    "title":       title,
                    "type":        inc_type,
                    "severity":    severity,
                    "imageUrl":    frame_to_b64(frame),
                    "location":    CAMERA_NAME,
                    "department":  "ایمنی و HSE",
                    "incidentDate":datetime.now().isoformat(),
                    "description": f"شناسایی هوشمند ({CONFIRM_FRAMES} فریم تأیید): {title}"
                }
                r = requests.post(API_URL, json=payload, verify=False, timeout=5)
                if r.status_code in (200, 201):
                    print(f"✅ ثبت شد در API: {title}")
                    last_alert_time       = now
                    active_incident_title = title
                else:
                    print(f"⚠️ API Status Code: {r.status_code}")
            except Exception as e:
                print(f"❌ خطا در ارسال: {e}")
    else:
        if now - last_alert_time > COOLDOWN_SECONDS:
            active_incident_title = None

    # نمایش وضعیت روی فریم
    status = f"HSE | {len(persons)}p {len(vehicles)}v | {datetime.now():%H:%M:%S}"
    cv2.putText(frame, status, (10, 25),
               cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,200), 1)

    time.sleep(0.03)

cap.release()
print("✅ پردازش تمام شد")