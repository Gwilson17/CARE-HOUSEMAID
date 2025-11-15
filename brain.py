from flask import Flask, request, send_file, render_template
import os

app = Flask(__name__)

FRAME_PATH = "frame.jpg"
status_data = {
    "user_detected": False,
    "fall_detected": False
}

@app.route("/")
def dash():
    return render_template("dashboard.html",
                           user_detected=status_data["user_detected"],
                           fall_detected=status_data["fall_detected"])

@app.route("/upload_frame", methods=["POST"])
def upload_frame():
    global status_data

    # save image
    file = request.files.get("frame")
    if file:
        file.save(FRAME_PATH)

    # get statuses
    user = request.form.get("user_detected", "false") == "true"
    fall = request.form.get("fall_detected", "false") == "true"

    status_data["user_detected"] = user
    status_data["fall_detected"] = fall

    return "OK", 200

@app.route("/frame.jpg")
def send_frame():
    if not os.path.exists(FRAME_PATH):
        return send_file("blank.jpg")
    return send_file(FRAME_PATH)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
