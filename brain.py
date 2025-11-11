import time
import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, request, jsonify, render_template, redirect, url_for
from datetime import datetime, timedelta
import base64
import threading
import smtplib
from email.message import EmailMessage
import google.auth
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

app = Flask(__name__)

# --- Global Variables ---
latest_frame = None
user_sleeping = False
user_in_bed = False
robot_command = "stop"
user_face_detected = False
user_image = None
last_seen_time = datetime.now()
alert_triggered = False
RECIPIENT_EMAIL = None  # Set dynamically

# --- OAuth2 Email Config ---
SENDER_EMAIL = "your_email@gmail.com"  # Gmail account
CREDENTIALS_FILE = "credentials.json"  # OAuth2 credentials JSON from Google Cloud

def send_email_oauth2(subject, body):
    """Send email via Gmail OAuth2."""
    global RECIPIENT_EMAIL
    if not RECIPIENT_EMAIL:
        print("⚠️ No recipient email set. Skipping email.")
        return
    try:
        creds = Credentials.from_authorized_user_file(CREDENTIALS_FILE, ["https://www.googleapis.com/auth/gmail.send"])
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
        
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = SENDER_EMAIL
        msg["To"] = RECIPIENT_EMAIL
        msg.set_content(body)

        import smtplib
        smtp_conn = smtplib.SMTP("smtp.gmail.com", 587)
        smtp_conn.ehlo()
        smtp_conn.starttls()
        smtp_conn.ehlo()
        smtp_conn.login(SENDER_EMAIL, creds.token)
        smtp_conn.send_message(msg)
        smtp_conn.quit()
        print(f"📧 Email sent via OAuth2: {subject} to {RECIPIENT_EMAIL}")
    except Exception as e:
        print("❌ Failed to send email via OAuth2:", e)

# --- MediaPipe Setup ---
mp_pose = mp.solutions.pose
mp_face_detection = mp.solutions.face_detection
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)

# ------------------------- ROUTES -------------------------
@app.route('/')
def index():
    global latest_frame, robot_command, user_in_bed, user_face_detected
    status = "Sleeping" if user_in_bed else "Awake"
    image_base64 = None
    if latest_frame is not None:
        _, buf = cv2.imencode('.jpg', latest_frame)
        image_base64 = base64.b64encode(buf).decode('utf-8')
    return render_template('dashboard.html',
                           status=status,
                           command=robot_command,
                           user_face_detected=user_face_detected,
                           image_base64=image_base64)

@app.route('/upload_initial_image', methods=['POST'])
def upload_initial_image():
    global latest_frame, user_face_detected, user_image, user_in_bed, robot_command, last_seen_time, alert_triggered

    if 'image' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    latest_frame = frame

    analyze_frame(frame)

    if user_face_detected:
        last_seen_time = datetime.now()
        alert_triggered = False
        user_image = frame

    return redirect(url_for('index'))

@app.route('/cmd', methods=['GET'])
def get_command():
    global robot_command
    return jsonify({"command": robot_command})

@app.route('/status', methods=['GET'])
def get_status():
    global latest_frame, robot_command, user_in_bed, user_face_detected
    status = "Sleeping" if user_in_bed else "Awake"
    image_base64 = None
    if latest_frame is not None:
        _, buf = cv2.imencode('.jpg', latest_frame)
        image_base64 = base64.b64encode(buf).decode('utf-8')
    return jsonify({
        "status": status,
        "command": robot_command,
        "user_face_detected": user_face_detected,
        "image_base64": image_base64,
        "timestamp": datetime.now().isoformat()
    })

@app.route('/set_sleep_mode', methods=['POST'])
def set_sleep_mode():
    global user_sleeping
    user_sleeping = not user_sleeping
    print(f"💤 Sleep mode set to: {user_sleeping}")
    return ('', 204)

@app.route('/set_email', methods=['POST'])
def set_email():
    global RECIPIENT_EMAIL
    email = request.form.get('alert_email')
    if email:
        RECIPIENT_EMAIL = email
        print(f"✅ Alert email set to: {RECIPIENT_EMAIL}")
    return redirect(url_for('index'))

# ------------------------- FRAME ANALYSIS -------------------------
def analyze_frame(frame):
    global user_in_bed, robot_command, user_face_detected

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_results = face_detection.process(rgb_frame)
    pose_results = pose.process(rgb_frame)

    if face_results.detections:
        user_face_detected = True
        robot_command = "follow"

        if pose_results.pose_landmarks:
            landmarks = pose_results.pose_landmarks.landmark
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]

            lying_down = abs(left_shoulder.y - left_hip.y) < 0.05

            if lying_down and not user_sleeping:
                user_in_bed = True
                robot_command = "alert_fall"
                print("⚠️ Fall detected!")
                send_email_oauth2(
                    "⚠️ Fall Detected",
                    "A fall has been detected. Please check on the user immediately."
                )
            else:
                user_in_bed = lying_down
        else:
            user_in_bed = False
    else:
        user_face_detected = False
        user_in_bed = False
        robot_command = "search"

# ------------------------- MISSING MONITOR -------------------------
def monitor_missing_user():
    global last_seen_time, alert_triggered
    while True:
        if not alert_triggered and (datetime.now() - last_seen_time) > timedelta(minutes=20):
            alert_triggered = True
            print("🚨 User missing for more than 20 minutes!")
            send_email_oauth2(
                "🚨 User Missing Alert",
                "The user has not been detected for over 20 minutes. Please check immediately."
            )
        time.sleep(60)

# ------------------------- MAIN -------------------------
if __name__ == '__main__':
    threading.Thread(target=monitor_missing_user, daemon=True).start()
    app.run(host='0.0.0.0', port=5000)
