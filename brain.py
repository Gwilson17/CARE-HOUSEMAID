import time
import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, request, jsonify, render_template, redirect, url_for, Response
from datetime import datetime, timedelta
import threading
import base64
import smtplib
from email.message import EmailMessage
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

# ------------------------- FLASK SETUP -------------------------
app = Flask(__name__)

# ------------------------- GLOBAL VARIABLES -------------------------
latest_frame = None
user_face_detected = False
robot_command = "stop"
sleep_mode = False
alert_email = None
user_in_bed = False
last_seen_time = datetime.now()
alert_triggered = False
user_x_pos = 0.5

# ------------------------- MEDIA PIPE SETUP -------------------------
mp_pose = mp.solutions.pose
mp_face_detection = mp.solutions.face_detection
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)

# ------------------------- GOOGLE OAUTH2 CONFIG -------------------------
SENDER_EMAIL = "your_email@gmail.com"        # Gmail account
CREDENTIALS_FILE = "credentials.json"       # OAuth2 credentials
TOKEN_FILE = "token.json"                   # OAuth2 token file

def send_email_oauth2(subject, body):
    """Send email via Gmail OAuth2"""
    global alert_email
    if not alert_email:
        print("⚠️ No recipient email set. Skipping email.")
        return
    try:
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, ["https://www.googleapis.com/auth/gmail.send"])
        if creds.expired and creds.refresh_token:
            creds.refresh(Request())
        
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = SENDER_EMAIL
        msg["To"] = alert_email
        msg.set_content(body)

        # Gmail SMTP using OAuth2 token
        smtp_conn = smtplib.SMTP("smtp.gmail.com", 587)
        smtp_conn.ehlo()
        smtp_conn.starttls()
        smtp_conn.ehlo()
        smtp_conn.login(SENDER_EMAIL, creds.token)
        smtp_conn.send_message(msg)
        smtp_conn.quit()
        print(f"📧 Email sent via OAuth2: {subject} to {alert_email}")
    except Exception as e:
        print("❌ Failed to send email via OAuth2:", e)

# ------------------------- ROUTES -------------------------
@app.route('/')
def index():
    status = "Sleeping" if sleep_mode else "Awake"
    return render_template(
        'dashboard.html',
        status=status,
        command=robot_command,
        user_face_detected=user_face_detected
    )

@app.route('/upload_initial_image', methods=['POST'])
def upload_initial_image():
    global latest_frame
    if 'image' not in request.files:
        return "No image uploaded", 400
    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    latest_frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    analyze_frame(latest_frame)
    return redirect(url_for('index'))

@app.route('/set_email', methods=['POST'])
def set_email():
    global alert_email
    email = request.form.get('alert_email')
    if email:
        alert_email = email
        print(f"✅ Alert email set: {alert_email}")
    return redirect(url_for('index'))

@app.route('/set_sleep_mode', methods=['POST'])
def toggle_sleep():
    global sleep_mode
    sleep_mode = not sleep_mode
    print(f"🔔 Sleep mode toggled: {sleep_mode}")
    return jsonify({"sleep_mode": sleep_mode})

@app.route('/set_command/<command>', methods=['POST'])
def set_command(command):
    global robot_command
    robot_command = command
    print(f"🤖 Robot command set: {robot_command}")
    return jsonify({"command": robot_command})

@app.route('/live_feed')
def live_feed():
    global latest_frame
    if latest_frame is None:
        return "No video feed", 404
    _, jpeg = cv2.imencode('.jpg', latest_frame)
    return Response(jpeg.tobytes(), mimetype='image/jpeg')

@app.route('/cmd', methods=['GET'])
def get_command():
    return jsonify({
        "command": robot_command,
        "x_pos": user_x_pos
    })

# ------------------------- FRAME ANALYSIS -------------------------
def analyze_frame(frame):
    """Detect face/body and update robot command"""
    global user_face_detected, user_in_bed, robot_command, user_x_pos, last_seen_time

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

        lying_down = abs(left_shoulder.y - left_hip.y) < 0.05
        if lying_down:
            user_in_bed = True
            robot_command = "alert_fall"
            send_email_oauth2("⚠️ Fall Detected", "A fall has been detected. Please check immediately.")
        else:
            user_in_bed = False
            robot_command = "follow"

    if face_results.detections:
        user_face_detected = True
        robot_command = "follow"
        last_seen_time = datetime.now()

    if not body_detected and not user_face_detected:
        robot_command = "search"
        user_in_bed = False

# ------------------------- MONITOR MISSING USER -------------------------
def monitor_missing_user():
    global last_seen_time, alert_triggered
    while True:
        if not alert_triggered and (datetime.now() - last_seen_time) > timedelta(minutes=20):
            alert_triggered = True
            print("🚨 User missing for 20+ minutes!")
            send_email_oauth2("🚨 User Missing Alert", "The user has not been detected for over 20 minutes. Please check immediately.")
        time.sleep(60)

# ------------------------- MAIN -------------------------
if __name__ == '__main__':
    threading.Thread(target=monitor_missing_user, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, debug=True)
