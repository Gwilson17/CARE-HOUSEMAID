import time
import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, request, jsonify, render_template, redirect, url_for
from datetime import datetime, timedelta
import base64
import threading
import smtplib

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

# --- Email Config ---
SENDER_EMAIL = "your_email@gmail.com"
SENDER_PASSWORD = "your_app_password"  # Use App Password for Gmail
RECIPIENT_EMAIL = "relative_email@gmail.com"

# --- MediaPipe Setup ---
mp_pose = mp.solutions.pose
mp_face_detection = mp.solutions.face_detection
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)

# ------------------------- EMAIL ALERT FUNCTION -------------------------
def send_email(subject, message):
    """Send an email alert."""
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        body = f"Subject:{subject}\n\n{message}"
        server.sendmail(SENDER_EMAIL, RECIPIENT_EMAIL, body)
        server.quit()
        print(f"📧 Email sent: {subject}")
    except Exception as e:
        print("❌ Failed to send email:", e)

# ------------------------- ROUTES -------------------------

@app.route('/')
def index():
    """Render dashboard HTML."""
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
    """Receive image from form and analyze it."""
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
    """Return current robot command."""
    global robot_command
    return jsonify({"command": robot_command})

@app.route('/status', methods=['GET'])
def get_status():
    """Return current status + image."""
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
    """Toggle sleep mode."""
    global user_sleeping
    user_sleeping = not user_sleeping
    return ('', 204)  # No content

# ------------------------- FRAME ANALYSIS -------------------------

def analyze_frame(frame):
    """Detect face + posture (for fall)."""
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

            # Fall detection
            if abs(left_shoulder.y - left_hip.y) < 0.05:
                user_in_bed = True
                robot_command = "alert_fall"
                print("⚠️ Fall detected!")
                send_email(
                    "⚠️ Fall Detected",
                    "A fall has been detected. Please check on the user immediately."
                )
            else:
                user_in_bed = False
        else:
            user_in_bed = False
    else:
        user_face_detected = False
        robot_command = "search"

# ------------------------- MISSING MONITOR -------------------------

def monitor_missing_user():
    """Check if user has been missing for 20+ minutes."""
    global last_seen_time, alert_triggered
    while True:
        if not alert_triggered and (datetime.now() - last_seen_time) > timedelta(minutes=20):
            alert_triggered = True
            print("🚨 User missing for more than 20 minutes!")
            send_email(
                "🚨 User Missing Alert",
                "The user has not been detected for over 20 minutes. Please check immediately."
            )
        time.sleep(60)

# ------------------------- MAIN -------------------------

if __name__ == '__main__':
    threading.Thread(target=monitor_missing_user, daemon=True).start()
    app.run(host='0.0.0.0', port=5000)
