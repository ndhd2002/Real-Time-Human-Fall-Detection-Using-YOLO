import os
import sys
import json
import signal
import datetime
import multiprocessing
from multiprocessing import Process
from utils import *
import cv2
from threading import Thread
import redis
import numpy as np
import time
from config.loader import cfg, CAMERAS, STREAM_MAXLEN, REDIS_HOSTNAME, REDIS_PORT
from logs.log_handler import logger


# REDIS_HOSTNAME = "localhost"
# CAMERAS = CAMERAS[:1]



def signal_handler(sig, frame):
    for process in multiprocessing.active_children():
        process.terminate()
    sys.exit(1)

signal.signal(signal.SIGINT, signal_handler)

conn = redis.Redis(host=REDIS_HOSTNAME, port=REDIS_PORT)
def calculate_gamma_from_histogram(frame):
    # Convert frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Calculate histogram
    hist = cv2.calcHist([gray_frame], [0], None, [256], [0,256])
    # Calculate cumulative distribution function
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf[-1]
    # Find the index where 90% of the pixels are darker
    target_percentile = 0.8
    target_value = np.argmax(cdf_normalized >= target_percentile)
    # Calculate gamma value
    gamma = np.log(target_value / 255) / np.log(0.5)
    return gamma


def adjust_image_gamma_lookuptable(image, gamma=1.0):
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    table = np.array([((i / 255.0) ** gamma) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def add_frames(camera_info, cfg):
    camera_id = camera_info['id']
    cap = cv2.VideoCapture(camera_info["rtsp"])
    # cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)  # Set timeout 5s
    # cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
    # fr_count = 0
    # tt_count = 0
    # start_time = time.time()
    # period_time = time.time()
    logger.debug(f'Starting process to read camera id: {camera_id}')

    while True:
        try:
            ret, frame = cap.read()
            if not ret:
            # re-read camera since occur error
                cap.release()
                cap = cv2.VideoCapture(camera_info['rtsp'])
                logger.debug(f'Read camera: {camera_id} error!')
                continue
            # gamma_frame = calculate_gamma_from_histogram(frame)
            # gamma = (1/gamma_frame)
            # frame = adjust_image_gamma_lookuptable(frame, gamma)
            frame = frame[::3, ::3]     
            start_time=time.time()
            frame_to_redis = serialize_img(frame)
            current_datetime = datetime.datetime.now()
            frame_info = {"time": str(current_datetime),
                          "starttime": start_time}
            result = {"frame": frame_to_redis,
                      "frame_info": json.dumps(frame_info)}
            conn.xadd(f"cam:{camera_info['id']}", result, maxlen=STREAM_MAXLEN) 
            time.sleep(0.02)
        except Exception as e:
            print(e)

def listening_update_info(conn,chanel):
    global PROCESS
    logger.debug("Init listening_update_info")
    pubsub = conn.pubsub()
    pubsub.subscribe(chanel)
    for message in pubsub.listen():
        if message['type'] == 'subscribe': 
            continue
        logger.debug(f"Receive message succesfull!, data is: {message}")
        data = json.loads(message['data'])
        if data["type"] == "camera":
            camera_id = data['id']
            if data["action"] == "add":
                p = Process(target=add_frames, args=(data, cfg))
                p.start()
                PROCESS[camera_id] = p
                # logger.debug(PROCESS)
            elif data["action"] == "update":
                #terminal process
                PROCESS[camera_id].terminate()
                del PROCESS[camera_id]
                conn.delete(f"camera:{camera_id}")
                #add process
                p = Process(target=add_frames, args=(data, cfg))
                p.start()
                PROCESS[camera_id] = p
                logger.debug(PROCESS)
            elif data["action"] == "delete":
                PROCESS[camera_id].terminate()
                del PROCESS[camera_id]
                conn.delete(f"camera:{camera_id}")
if __name__ == "__main__":
    gate_info = {
        "gate_1": {
            "interval_time":10,
            "gate_name" : "GATE_1",
            "camera" : [
                {"id": "CAM1",
                "rtsp": "rtsp://rtsp_server:8554/cam1"
                # "rtsp": "C:\Users\ndhdu\Downloads\fall_detect\camera_service\camera1.mp4"
                },
                # {"id": "CAM2",
                # # "rtsp": "rtsp://rtsp_server:8554/cam2"
                # "rtsp": "camera2_reid.mp4"
                # }
                ]
            }
        }
    PROCESS = {}
    for gate_id,data_gate_info in gate_info.items():
        for camera_info in data_gate_info["camera"]:
            p = Process(target=add_frames, args=(camera_info , cfg))
            p.start()
            PROCESS[camera_info["id"]] = p
    chanel = 'update_info'
    Thread(target = listening_update_info, args=(conn,chanel)).start()