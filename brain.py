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
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
import serial
import json

# ------------------------- FLASK APP SETUP -------------------------
app = Flask(__name__)

# ------------------------- GLOBAL VARIABLES -------------------------
latest_frame = None
user_sleeping = False
user_in_bed = False
robot_command = "stop"
user_face_detected = False
user_image = None
last_seen_time = datetime.now()
alert_triggered = False
RECIPIENT_EMAIL = None
user_x_pos = 0.5
ultrasonic_distance = None

# ------------------------- ARDUINO SERIAL SETUP -------------------------
try:
    # Change COM port to match your Arduino connection (e.g. "COM4" or "/dev/ttyUSB0")
    arduino = serial.Serial("COM3", 9600, timeout=1)
    print("✅ Connected to Arduino on COM3")
except Exception as e:
    arduino = None
    print("⚠️ Arduino not connected:", e)

# ------------------------- EMAIL CONFIG -------------------------
SENDER_EMAIL = "vaargv23@gmail.com"
CREDENTIALS_FILE = "credentials.json"  # Gmail OAuth2 credentials file


def send_email_oauth2(subject, body):
    """Send an alert email via Gmail OAuth2."""
    global RECIPIENT_EMAIL
    if not RECIPIENT_EMAIL:
        print("⚠️ No recipient email set.")
        return
    try:
        creds = Credentials.from_authorized_user_file(
            CREDENTIALS_FILE, ["https://www.googleapis.com/auth/gmail.send"]
        )
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())

        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = SENDER_EMAIL
        msg["To"] = RECIPIENT_EMAIL
        msg.set_content(body)

        smtp_conn = smtplib.SMTP("smtp.gmail.com", 587)
        smtp_conn.starttls()
        smtp_conn.login(SENDER_EMAIL, creds.token)
        smtp_conn.send_message(msg)
        smtp_conn.quit()
        print(f"📧 Email sent: {subject}")
    except Exception as e:
        print("❌ Email send failed:", e)


# ------------------------- MEDIAPIPE SETUP -------------------------
mp_pose = mp.solutions.pose
mp_face_detection = mp.solutions.face_detection
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)

# ------------------------- ROUTES -------------------------
@app.route('/')
def index():
    """Main dashboard."""
    global latest_frame, robot_command, user_in_bed, user_face_detected, ultrasonic_distance
    status = "Sleeping" if user_in_bed else "Awake"
    distance = ultrasonic_distance if ultrasonic_distance else "N/A"
    image_base64 = None

    if latest_frame is not None:
        _, buf = cv2.imencode('.jpg', latest_frame)
        image_base64 = base64.b64encode(buf).decode('utf-8')

    return render_template(
        'dashboard.html',
        status=status,
        command=robot_command,
        distance=distance,
        user_face_detected=user_face_detected,
        image_base64=image_base64
    )


@app.route('/set_email', methods=['POST'])
def set_email():
    """Set email for alerts."""
    global RECIPIENT_EMAIL
    email = request.form.get('alert_email')
    if email:
        RECIPIENT_EMAIL = email
        print(f"✅ Alert email set: {RECIPIENT_EMAIL}")
    return redirect(url_for('index'))


@app.route('/upload_image', methods=['POST'])
def upload_image():
    """ESP32-CAM uploads image here."""
    global latest_frame, user_face_detected, user_image, user_in_bed, robot_command, last_seen_time, alert_triggered

    if 'image' not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    latest_frame = frame

    analyze_frame(frame)

    if user_face_detected:
        last_seen_time = datetime.now()
        alert_triggered = False
        user_image = frame

    return jsonify({"message": "Image received", "command": robot_command})


@app.route('/cmd', methods=['GET'])
def get_command():
    """Arduino can request current command."""
    global robot_command, user_x_pos
    return jsonify({"command": robot_command, "x_pos": user_x_pos})


@app.route('/sensor', methods=['POST'])
def update_sensor_data():
    """Arduino posts ultrasonic readings."""
    global ultrasonic_distance
    data = request.json
    ultrasonic_distance = data.get("distance")
    print(f"📡 Ultrasonic: {ultrasonic_distance} cm")

    if ultrasonic_distance and ultrasonic_distance < 15:
        # Obstacle close — stop motors
        robot_command = "stop"
        print("🛑 Obstacle detected — stopping motors")

    return jsonify({"status": "ok"})


# ------------------------- FRAME ANALYSIS -------------------------
def analyze_frame(frame):
    """Analyze incoming frame from ESP32-CAM."""
    global user_in_bed, robot_command, user_face_detected, user_x_pos, user_sleeping

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_results = face_detection.process(rgb_frame)
    pose_results = pose.process(rgb_frame)

    user_x_pos = 0.5
    user_face_detected = False
    body_detected = False

    if pose_results.pose_landmarks:
        body_detected = True
        landmarks = pose_results.pose_landmarks.landmark
        left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
        user_x_pos = (left_shoulder.x + right_shoulder.x) / 2

        # Check if lying down
        lying_down = abs(left_shoulder.y - left_hip.y) < 0.05
        if lying_down and not user_sleeping:
            user_in_bed = True
            robot_command = "alert_fall"
            print("⚠️ Fall detected!")
            send_email_oauth2(
                "⚠️ Fall Detected",
                "A fall has been detected. Please check immediately."
            )
        else:
            user_in_bed = False
            robot_command = "follow"

    if face_results.detections:
        user_face_detected = True
        robot_command = "follow"

    if not body_detected and not user_face_detected:
        robot_command = "search"
        user_in_bed = False


# ------------------------- BACKGROUND THREADS -------------------------
def monitor_missing_user():
    """Check if user has been missing for > 20 min."""
    global last_seen_time, alert_triggered
    while True:
        if not alert_triggered and (datetime.now() - last_seen_time) > timedelta(minutes=20):
            alert_triggered = True
            print("🚨 User missing for >20 minutes!")
            send_email_oauth2(
                "🚨 User Missing",
                "The user has not been detected for over 20 minutes."
            )
        time.sleep(60)


def serial_command_sender():
    """Send updated robot_command to Arduino continuously."""
    global robot_command
    last_cmd = ""
    while True:
        if arduino and robot_command != last_cmd:
            try:
                arduino.write((robot_command + "\n").encode())
                print(f"📤 Sent to Arduino: {robot_command}")
                last_cmd = robot_command
            except Exception as e:
                print("⚠️ Serial send error:", e)
        time.sleep(1)


def serial_listener():
    """Optional: read data coming from Arduino (like ultrasonic)."""
    global ultrasonic_distance
    if not arduino:
        return
    while True:
        try:
            if arduino.in_waiting:
                line = arduino.readline().decode().strip()
                if line.startswith("{") and line.endswith("}"):
                    data = json.loads(line)
                    ultrasonic_distance = data.get("distance", ultrasonic_distance)
                    print(f"📩 From Arduino: {data}")
        except Exception as e:
            print("⚠️ Serial read error:", e)
        time.sleep(0.1)


# ------------------------- MAIN -------------------------
if __name__ == '__main__':
    threading.Thread(target=monitor_missing_user, daemon=True).start()
    threading.Thread(target=serial_command_sender, daemon=True).start()
    threading.Thread(target=serial_listener, daemon=True).start()
    app.run(host='0.0.0.0', port=5000)
