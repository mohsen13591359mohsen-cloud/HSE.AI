import cv2
import numpy as np
from ultralytics import YOLO
import requests
import time
import base64
from datetime import datetime
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ═══════════════════════════════════════════════════════════
# ⚙️ تنظیمات و آدرس ngrok
# ═══════════════════════════════════════════════════════════
BASE_NGROK_URL   = "https://outfit-dimly-juice.ngrok-free.dev"
API_URL          = f"{BASE_NGROK_URL}/api/violations/camera"

CAMERA_NAME      = "دوربین ۲ - خط تولید A"
VIDEO_SOURCE     = "vid2.mp4"
COOLDOWN_SECONDS = 30
CONF_THRESHOLD   = 0.50   # حداقل درصد اطمینان برای تشخیص شخص
CONFIRM_FRAMES   = 3      # تعداد فریم متوالی جهت تأیید واقعی تخلف

# ═══════════════════════════════════════════════════════════
# 🧠 بارگذاری مدل استاندارد و آماده YOLO
# ═══════════════════════════════════════════════════════════
print("=" * 70)
print("👝 موتور هوشمند تشخیص عدم استفاده از کلاه ایمنی (PPE) فعال شد...")
print("=" * 70)

# مدل رسمی ultralytics بدون نیاز به لینک‌های متفرقه
model = YOLO("yolov8s.pt")

cap = cv2.VideoCapture(VIDEO_SOURCE)

last_alert_time     = 0
is_violation_active = False
confirm_counter     = 0

def convert_frame_to_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{jpg_as_text}"

def has_helmet(head_crop):
    """
    بررسی وجود کلاه ایمنی در ناحیه سر شخص بر اساس بازه رنگی HSV
    کلاه‌های رایج ایمنی: زرد، سفید، قرمز، آبی، نارنجی
    """
    if head_crop.size == 0:
        return True # برای جلوگیری از خطای کادر خالی

    hsv = cv2.cvtColor(head_crop, cv2.COLOR_BGR2HSV)
    
    # محدوده رنگ زرد و نارنجی (کلاه‌های ایمنی رایج)
    yellow_lower = np.array([15, 100, 100])
    yellow_upper = np.array([35, 255, 255])
    
    # محدوده رنگ سفید (کلاه‌های ایمنی مهندسی)
    white_lower = np.array([0, 0, 180])
    white_upper = np.array([180, 40, 255])

    # محدوده رنگ آبی
    blue_lower = np.array([90, 80, 80])
    blue_upper = np.array([130, 255, 255])

    # محدوده رنگ قرمز
    red_lower1 = np.array([0, 100, 100])
    red_upper1 = np.array([10, 255, 255])
    red_lower2 = np.array([160, 100, 100])
    red_upper2 = np.array([179, 255, 255])

    mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
    mask_white  = cv2.inRange(hsv, white_lower, white_upper)
    mask_blue   = cv2.inRange(hsv, blue_lower, blue_upper)
    mask_red    = cv2.bitwise_or(cv2.inRange(hsv, red_lower1, red_upper1), cv2.inRange(hsv, red_lower2, red_upper2))

    combined_mask = mask_yellow | mask_white | mask_blue | mask_red
    
    # نسبت پیکسل‌های کلاه به کل ناحیه سر
    helmet_pixel_ratio = cv2.countNonZero(combined_mask) / float(head_crop.shape[0] * head_crop.shape[1])
    
    # اگر بیش از ۱۵ درصد ناحیه سر رنگ کلاه داشته باشد، کلاه دارد
    return helmet_pixel_ratio > 0.15

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        is_violation_active = False
        confirm_counter = 0
        continue

    results = model(frame, conf=CONF_THRESHOLD, verbose=False)
    current_frame_has_violation = False

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            conf = float(box.conf[0])
            
            # فقط پردازش اشخاص
            if class_name == "person":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # ۱. جدا کردن ۲۰٪ بالای کادر (محدوده سر انسان)
                head_h = int((y2 - y1) * 0.25)
                head_crop = frame[y1:y1 + head_h, x1:x2]
                
                # ۲. چک کردن کلاه
                if not has_helmet(head_crop):
                    current_frame_has_violation = True
                    
                    # رسم کادر قرمز دور شخص بدون کلاه
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"No Helmet! ({conf:.0%})", 
                                (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    # رسم کادر سبز دور شخص با کلاه
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)

    # سیستم تأیید ۳ فریمی
    if current_frame_has_violation:
        confirm_counter = min(confirm_counter + 1, CONFIRM_FRAMES + 1)
    else:
        confirm_counter = max(confirm_counter - 1, 0)

    violation_confirmed = (confirm_counter >= CONFIRM_FRAMES)
    current_time = time.time()

    # ارسال به API
    if violation_confirmed:
        if not is_violation_active and (current_time - last_alert_time > COOLDOWN_SECONDS):
            alert_name = "عدم استفاده از کلاه ایمنی در خط تولید"
            try:
                image_base64 = convert_frame_to_base64(frame)
                payload = {
                    "violationType": alert_name,
                    "imageUrl": image_base64,
                    "cameraLocation": CAMERA_NAME,
                    "detectedAt": datetime.now().isoformat(),
                    "hseComment": f"شناسایی خودکار عدم استفاده از کلاه ایمنی روی {CAMERA_NAME}"
                }
                
                response = requests.post(API_URL, json=payload, verify=False, timeout=5)
                
                if response.status_code in [200, 201]:
                    print(f"🎯 [{datetime.now():%H:%M:%S}] تخلف عدم کلاه ثبت شد!")
                    last_alert_time = current_time
                    is_violation_active = True
                elif response.status_code == 502:
                    print("⚠️ خطای 502: پروژه دات‌نت لوکال متصل نیست یا ngrok قطع است.")
                else:
                    print(f"⚠️ پاسخ دات‌نت: {response.status_code}")
                    
            except Exception as e:
                print(f"❌ خطا در ارسال داده به API: {e}")
    else:
        if current_time - last_alert_time > COOLDOWN_SECONDS:
            is_violation_active = False

    time.sleep(0.03)

cap.release()