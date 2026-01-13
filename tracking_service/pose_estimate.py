import os
import sys
import json 
from typing import List 
from dataclasses import dataclass
import time
import cv2
import numpy as np
from ultralytics import YOLO
from config.loader import STREAMMAXLEN, MODEL
from logs.log_handler import logger
from reID import ReIDManager

class PoseEstimator:
    def __init__(self, cam_id):
        self.model = YOLO(MODEL)  # Ensure MODEL points to a YOLOv8 pose model
        self.starttime = time.time()
        self.model.fuse()
        self.redis_key = f"cam:{cam_id}"
        self.cam_id = cam_id
        self.tracked_key = f"tracked_cam:{cam_id}"
        self.previous_index = None
        self.fps_window = []
        self.fps_window_size = 10 
        self.reid = ReIDManager()
        self.next_id = 0  # Global ID counter

    def estimate_pose(self, frame, count):
        current_time = time.time()
        results = self.model.predict(
            frame,
            conf=0.7,
            iou=0.5,
            verbose=False
        )
        process_time = time.time()
        frame_duration = process_time - current_time
        instant_fps = 1.0 / frame_duration if frame_duration > 0 else 0

        # FPS
        self.fps_window.append(instant_fps)
        if len(self.fps_window) > self.fps_window_size:
            self.fps_window.pop(0)

        smoothed_fps = sum(self.fps_window) / len(self.fps_window)


        if not results or results[0].boxes is None or results[0].keypoints is None:
            return []

        detected_objects = []
        boxes = results[0].boxes.data.cpu().numpy()
        keypoints_data = results[0].keypoints.data.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()

        for idx, (box, kps, cls_id, conf) in enumerate(zip(boxes, keypoints_data, classes, confs)):
            x, y, w, h = box[:4]
            x, y, w, h = int(x), int(y), int(w), int(h)
            bbox = [x, y, w, h]
            keypoints_2d = kps[:, :2].tolist()
            keypoints_conf = kps[:, 2].tolist()

            # ReID
            person_img = frame[y:y+h, x:x+w]
            if person_img.size == 0 or person_img.shape[0] == 0 or person_img.shape[1] == 0:
                continue

            embedding = self.reid.extract_embedding(person_img)
            global_id = self.reid.match_or_create_global_id(embedding)

            pose_data = {
                "id": idx,
                "cam_id": self.cam_id,
                "class": int(cls_id),
                "bbox": bbox,
                "bbox_conf": float(conf),
                "keypoints": keypoints_2d,
                "keypoints_conf": keypoints_conf,
                "status": " "
            }
            detected_objects.append(pose_data)


        return detected_objects, smoothed_fps

    def update(self, data, count):
        byte_frame = data[b"frame"]
        frame_info = json.loads(data[b"frame_info"])
        frame = cv2.imdecode(np.frombuffer(byte_frame, np.uint8), cv2.IMREAD_COLOR)
        pose_result, smoothed_fps = self.estimate_pose(frame, count)
        
        frame_info["objects"] = pose_result
        frame_info["fps"] = smoothed_fps

        result = {
            "frame": byte_frame,
            "frame_info": json.dumps(frame_info)
        }

        return result

    def get_data(self, conn):
        try:
            p = conn.pipeline()
            p.xrevrange(self.redis_key, count=1)
            msg = p.execute()

            index = None
            if msg and len(msg[0]) > 0:
                index = msg[0][0][0].decode("utf-8")

            if ((index is None) or 
                (self.previous_index is not None and 
                 self.previous_index == index)):
                return None
            
            self.previous_index = index
            data = msg[0][0][1]
            return data
        
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return None
            
    def run(self, conn):
        count = 0
        while True:
            data = self.get_data(conn)
            if data:
                result = self.update(data, count)
                count += 1
                conn.xadd(self.tracked_key, result, maxlen=STREAMMAXLEN)
