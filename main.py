
import cv2
from ultralytics import YOLO
import requests
import time
import base64
from datetime import datetime
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# بارگذاری مدل
model = YOLO("yolov8n.pt")

video_source = "vid2.mp4"
cap = cv2.VideoCapture(video_source)

# 🎯 نکته مهم: آدرس ngrok سیستم خود را جایگزین کنید!
BASE_NGROK_URL = "https://a1b2-c3d4-e5f6.ngrok-free.app"  # آدرس اختصاصی شما
COOLDOWN_SECONDS = 30
API_URL = f"{BASE_NGROK_URL}/api/violations/camera"
CAMERA_NAME = "دوربین ۲ - خط تولید A"

last_alert_time = 0
is_person_in_frame = False

def convert_frame_to_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{jpg_as_text}"

print("🧠 موتور پردازش تصویر هوشمند برای دوربین ۲ در Colab فعال شد...")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        is_person_in_frame = False
        continue

    results = model(frame, verbose=False)
    current_frame_has_person = False
    alert_name = ""

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            
            if class_name == "person":
                current_frame_has_person = True
                alert_name = "عدم استفاده از کلاه ایمنی (تست)"
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    current_time = time.time()

    if current_frame_has_person:
        if not is_person_in_frame and (current_time - last_alert_time > COOLDOWN_SECONDS):
            try:
                image_base64 = convert_frame_to_base64(frame)
                payload = {
                    "violationType": alert_name,
                    "imageUrl": image_base64,
                    "cameraLocation": CAMERA_NAME,
                    "detectedAt": datetime.now().isoformat(),
                    "hseComment": f"شناسایی خودکار توسط هوش مصنوعی Colab روی {CAMERA_NAME}"
                }
                
                response = requests.post(API_URL, json=payload, verify=False, timeout=5)
                
                if response.status_code in [200, 201]:
                    print(f"🎯 [ثبت تک‌باره Colab] تخلف دوربین ۲ ثبت شد: {alert_name}")
                    last_alert_time = current_time
                    is_person_in_frame = True
                else:
                    print(f"⚠️ پاسخ دات‌نت: {response.status_code}")
                    
            except Exception as e:
                print(f"❌ خطا در ارسال داده به API: {e}")
    else:
        is_person_in_frame = False

    time.sleep(0.03)

cap.release()