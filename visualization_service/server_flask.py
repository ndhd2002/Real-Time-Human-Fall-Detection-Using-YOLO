import cv2
import redis
import numpy as np
import json
import time
import os
from flask import Flask, render_template, Response, jsonify
from supervision import ColorPalette
from config.loader import REDIS_HOSTNAME, REDIS_PORT, CAMERAS
from config.loader import (
    STREAMMAXLEN,
    THICKNESS,
    TEXTTHICKNESS,
    TEXTSCALE,
    TRIANGLESIZE,
    OFFSET,
)
from logs.log_handler import logger
from annotators import Annotator

r = redis.Redis(host=REDIS_HOSTNAME, port=REDIS_PORT)
app = Flask(__name__)

annotator = Annotator(
    color=ColorPalette(10),
    thickness=THICKNESS,
    text_thickness=TEXTTHICKNESS,
    text_scale=TEXTSCALE,
    anchor="center"
)

def drawing(frame, frame_info):
    objects = frame_info["objects"]
    fps=frame_info['fps']
    if objects:
        frame = annotator.annotate(
            scene=frame,
            detections=objects,
            fps=fps
        )
    return frame

def get_data(conn, cam_index):
    while True:
        p = conn.pipeline()
        p.xrevrange(f"fall_cam:{CAMERAS[cam_index]['id']}", count=4)
        msg = p.execute()

        if msg and msg[0]:
            data = msg[0][0][1]
            frame_data = data.get(b"frame")
            if not frame_data:
                continue

            frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
            frame_info = json.loads(data[b"frame_info"])

            frame = drawing(frame, frame_info)
            for obj in frame_info["objects"]:
                if obj.get("fall_detected"):
                    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
                    cam_id = CAMERAS[cam_index]['id']
                    filename = f"static/falls/{cam_id}_{timestamp}.jpg"
                    cv2.imwrite(filename, frame)
                    logger.info(f"Fall detected - saved: {filename}")
                    break 

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                logger.debug(f"Failed to encode frame from cam {CAMERAS[cam_index]['id']}")
                continue

            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            time.sleep(0.05)

def load_fall_events_all_cams():
    """
    Load all fall events from all cameras.
    Assumes images are saved as camId_YYYY-MM-DD_HH-MM-SS.jpg
    """
    fall_events = []
    fall_dir = os.path.join("static", "falls")

    if not os.path.exists(fall_dir):
        return []

    for fname in sorted(os.listdir(fall_dir), reverse=True):
        if fname.endswith(".jpg"):
            try:
                cam_id, time_str = fname.split("_", 1)
                time_display = time_str.replace(".jpg", "").replace("_", " ")
                fall_events.append({
                    "image_path": f"falls/{fname}",
                    "time": time_display,
                    "camera": cam_id
                })
            except ValueError:
                continue
    return fall_events

@app.route("/")
def index():
    fall_events = load_fall_events_all_cams()
    return render_template("index.html", fall_events=fall_events)

@app.route("/fall_events")
def fall_events():
    fall_events = load_fall_events_all_cams()
    return jsonify(fall_events)

@app.route("/video_feed_1")
def video_feed_1():
    return Response(get_data(r, 0), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/video_feed_2")
def video_feed_2():
    return Response(get_data(r, 1), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    app.run(debug=True, port=5000, host="0.0.0.0")
