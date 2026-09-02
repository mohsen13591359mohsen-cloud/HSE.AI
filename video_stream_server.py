%%writefile video_stream_server.py
import cv2
import time
from flask import Flask, Response
from flask_cors import CORS
import threading

app = Flask(__name__)
CORS(app)

VIDEO_SOURCE = "vid2.mp4"
TARGET_FPS   = 15
JPEG_QUALITY = 70

cap_lock     = threading.Lock()
current_cap  = None

def get_cap():
    global current_cap
    if current_cap is None or not current_cap.isOpened():
        current_cap = cv2.VideoCapture(VIDEO_SOURCE)
        current_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return current_cap

def generate_frames():
    frame_interval = 1.0 / TARGET_FPS
    while True:
        loop_start = time.time()
        try:
            with cap_lock:
                cap = get_cap()
                ok, frame = cap.read()
                if not ok:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = cap.read()
                    if not ok:
                        time.sleep(0.1)
                        continue

            frame = cv2.resize(frame, (854, 480))
            ts = time.strftime("%H:%M:%S")
            cv2.putText(frame, f"LIVE  {ts}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n'
                   + buf.tobytes()
                   + b'\r\n')

        except Exception as e:
            time.sleep(0.1)
            continue

        elapsed = time.time() - loop_start
        sleep_t = frame_interval - elapsed
        if sleep_t > 0:
            time.sleep(sleep_t)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/health')
def health():
    return {'status': 'ok', 'source': str(VIDEO_SOURCE)}

if __name__ == '__main__':
    print("🎥 Stream server آماده اجراست...")
    app.run(host='0.0.0.0', port=5005, threaded=True)