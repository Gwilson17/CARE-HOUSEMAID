import time
import cv2
import numpy as np
import mediapipe as mp
from flask import Flask, render_template, request, jsonify
from datetime import datetime
import base64
import smtplib
from email.mime.text import MIMEText

app = Flask(__name__)

# --- Global Variables ---
latest_frame = None
latest_distances = None
user_sleeping = False
user_in_bed = False
robot_command = "stop"  # Command to send to ESP32-CAM
user_face_detected = False
user_image = None

# --- MediaPipe Setup ---
mp_pose = mp.solutions.pose
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)

# --- Flask Routes ---
@app.route('/')
def index():
    """Render dashboard page."""
    status = "Sleeping" if user_in_bed else "Awake"
    return render_template('index.html', status=status, command=robot_command, user_face_detected=user_face_detected)

@app.route('/upload_initial_image', methods=['POST'])
def upload_initial_image():
    """Receive the initial image from the user to process."""
    global latest_frame, latest_distances, user_face_detected, user_image

    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    # Read image from the user
    img_bytes = request.files['image'].read()
    npimg = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    latest_frame = frame

    # Process the uploaded image to detect faces
    analyze_frame(frame)

    if user_face_detected:
        user_image = frame
        response = {"status": "Face detected, robot can follow", "robot_command": robot_command}
    else:
        response = {"status": "No face detected, robot is searching", "robot_command": robot_command}

    return jsonify(response)

@app.route('/set_sleep_mode', methods=['POST'])
def set_sleep_mode():
    """Toggle sleep mode"""
    global user_sleeping
    user_sleeping = not user_sleeping
    status = "Activated" if user_sleeping else "Deactivated"
    return jsonify({"status": f"Sleep mode {status}"})

# --- Flask Logic for Fall Detection, Robot Movement & Notification ---
def analyze_frame(frame):
    global user_sleeping, user_in_bed, robot_command, user_face_detected

    # Detect faces
    results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Check for face detection and pose tracking for fall detection
    if results.detections:
        user_face_detected = True
    else:
        user_face_detected = False

    # Simulate user status based on detection logic (for now, basic logic for user lying down or standing)
    # Placeholder for actual fall detection using pose
    if user_face_detected:
        # Additional logic for detecting whether user is standing, sitting, or lying down can go here
        user_in_bed = True
    else:
        user_in_bed = False

    if user_in_bed and not user_sleeping:
        robot_command = "stop"  # Stop robot if sleeping
    else:
        robot_command = "move"  # Continue following the user or perform other actions

@app.route('/status', methods=['GET'])
def get_status():
    """Return user status, command, and last image for dashboard auto-refresh."""
    global latest_frame, robot_command, user_in_bed

    status = "Sleeping" if user_in_bed else "Awake"
    image_base64 = None

    if latest_frame is not None:
        _, buf = cv2.imencode('.jpg', latest_frame)
        image_base64 = base64.b64encode(buf).decode('utf-8')

    return jsonify({
        "status": status,
        "command": robot_command,
        "image_base64": image_base64,
        "timestamp": datetime.now().isoformat()
    })

# --- Start Flask Server ---
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
