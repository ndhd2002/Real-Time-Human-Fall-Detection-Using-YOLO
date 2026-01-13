import os
import json
from datetime import datetime
import time
import numpy as np
from supervision import Point, Detections
from config.loader import STREAMMAXLEN
from logs.log_handler import logger


class Falling:
    def __init__(self, cam_id):
        self.tracked_key = f"tracked_cam:{cam_id}"
        self.prowled_key = f"fall_cam:{cam_id}" 
        self.json_fpath = f"{cam_id}.json"
        self.first_appear_time = {}
        self.cleanup_interval = 10  # seconds
        self.last_cleanup_time = time.time()
        self.global_falling_status = {}  # {id: bool}
        self.previous_index = None
        self.previous_centers = {}  # Store previous center points and time
        self.velocity_history = {}  # Store velocity history for each ID
        self.last_saved_times = {}  # Per-ID last saved time
        self.shoulder_history = {}  # Per-ID shoulder Y history

        if os.path.exists(self.json_fpath):
            with open(self.json_fpath, "r") as f:
                _ = json.load(f)  # placeholder in case of later use

    def cleanup_first_appear_time(self):
        current_time = time.time()
        if current_time - self.last_cleanup_time >= self.cleanup_interval:
            self.first_appear_time = {
                id: timestamp
                for id, timestamp in self.first_appear_time.items()
                if current_time - timestamp <= 10
            }
            self.last_cleanup_time = current_time

    def cleanup_velocity_history(self):
        current_time = time.time()
        self.velocity_history = {
            idx: (velocities, last_time)
            for idx, (velocities, last_time) in self.velocity_history.items()
            if current_time - last_time <= 30
        }
        self.last_saved_times = {
            idx: last_time
            for idx, last_time in self.last_saved_times.items()
            if current_time - last_time <= 30
        }
        self.shoulder_history = {
            idx: [(t, y) for t, y in points if current_time - t <= 2]
            for idx, points in self.shoulder_history.items()
        }
    
    def get_midpoint(self, pt1, pt2):
        return [(pt1[0] + pt2[0]) / 2, (pt1[1] + pt2[1]) / 2]

    # def calculate_angle(self, shoulders_mid, hips_mid):
    #     vector = np.array(shoulders_mid) - np.array(hips_mid)
    #     angle = np.degrees(np.arctan2(vector[1], vector[0]))
    #     return abs(angle)
    
    def save_velocity_history_to_json(self):
        current_time = time.time()
        history_to_save = {}
        for idx, (velocities, _) in self.velocity_history.items():
            last_save = self.last_saved_times.get(idx, 0)
            if current_time - last_save >= 1.0:
                history_to_save[idx] = velocities
                self.last_saved_times[idx] = current_time

        if history_to_save:
            try:
                with open(self.json_fpath, "r") as f:
                    existing_data = json.load(f)
            except:
                existing_data = {}

            existing_data.update(history_to_save)

            with open(self.json_fpath, "w") as f:
                json.dump(existing_data, f, indent=2)

    def calculate_center_of_8_points(self, points):
        return np.mean(points, axis=0).tolist()

    def calculate_velocity(self, idx, current_center, current_time):
        if idx in self.previous_centers:
            prev_center, prev_time = self.previous_centers[idx]
            time_diff = current_time - prev_time
            if time_diff > 0:
                displacement = np.linalg.norm(np.array(current_center) - np.array(prev_center))
                velocity = displacement / time_diff
            else:
                velocity = 0.0
        else:
            velocity = 0.0
        self.previous_centers[idx] = (current_center, current_time)

        if idx not in self.velocity_history:
            self.velocity_history[idx] = ([velocity], current_time)
        else:
            last_saved = self.velocity_history[idx][1]
            if current_time - last_saved >= 1.0:
                self.velocity_history[idx][0].append(velocity)
                self.velocity_history[idx] = (self.velocity_history[idx][0], current_time)

        return velocity

    def calculate_angle_to_vertical(self, center_8, ankle_mid):
        # The vector from the ankle to the body (since the y-axis goes down).
        vector = np.array(center_8) - np.array(ankle_mid)
        
        # The vertical direction pointing upwards in the image.
        vertical = np.array([0, -1])
        
        vector_norm = np.linalg.norm(vector)
        if vector_norm == 0:
            return 0.0

        cos_theta = np.dot(vector, vertical) / vector_norm
        angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
        return angle


    def detect_fall(self, idx, angle_to_vertical, shoulders_mid):
        if idx not in self.velocity_history:
            return False

        velocities, _ = self.velocity_history[idx]
        if len(velocities) < 3:
            return False

        max_velocity = max(velocities)
        if max_velocity == 0:
            return False

        threshold_high = 0.75 * max_velocity
        threshold_low = 0.25 * max_velocity

        if idx not in self.shoulder_history:
            self.shoulder_history[idx] = []

        self.shoulder_history[idx].append((time.time(), shoulders_mid[1]))

        shoulder_ys = [y for _, y in self.shoulder_history[idx]]

        for i in range(len(velocities) - 2):
            if velocities[i] >= 200 and velocities[i] > threshold_high and velocities[i + 1] < threshold_low:
                if angle_to_vertical > 50:
                    return True
                if len(shoulder_ys) >= 2:
                    delta_shoulder_y = shoulder_ys[-1] - shoulder_ys[0]
                    if delta_shoulder_y > 20:  # The shoulder drops significantly.
                        return True

        return False

    

    def data2result(self, data):
        frame_info = json.loads(data[b"frame_info"])
        objects = frame_info["objects"]
        self.cleanup_first_appear_time()
        self.cleanup_velocity_history()
        status_object = []

        frame_timestamp = frame_info["starttime"]

        for object in objects:
            bbox = object["bbox"]
            keypoints = object["keypoints"]
            idx = object["id"]
            if keypoints and len(keypoints) >= 8:
                if any(k is None for k in keypoints[:8]):
                    continue

                sd_l, sd_r = keypoints[0], keypoints[1]
                h_l, h_r = keypoints[2], keypoints[3]
                k_l, k_r = keypoints[4], keypoints[5]
                a_l, a_r = keypoints[6], keypoints[7]

                center = self.calculate_center_of_8_points([sd_l, sd_r, h_l, h_r, k_l, k_r, a_l, a_r])
                ankle_mid = self.get_midpoint(a_l, a_r)
                shoulders_mid = self.get_midpoint(sd_l, sd_r)
                velocity = self.calculate_velocity(idx, shoulders_mid, frame_timestamp)
                angle_to_vertical = self.calculate_angle_to_vertical(center, ankle_mid)

                is_falling = self.detect_fall(idx, angle_to_vertical, shoulders_mid)
                if self.global_falling_status.get(idx, False):
                    is_falling = True
                if is_falling:
                    self.global_falling_status[idx] = True

                # logger.debug(f"Object ID {idx} center: {center}, velocity: {velocity:.2f}")

                object = {
                    "id": idx,
                    "bbox": bbox,
                    "keypoints": keypoints,
                    "center": center,
                    "velocity": velocity,
                    "angle_to_vertical": angle_to_vertical,
                    "fall_detected": is_falling,
                }
                status_object.append(object)

        self.save_velocity_history_to_json()

        frame_info["objects"] = status_object
        # logger.debug(status_object)
        result = {
            "frame": data[b"frame"],
            "frame_info": json.dumps(frame_info)
        }
        return result


    def update(self, data):
        return self.data2result(data)

    def get_data(self, conn):
        try:
            p = conn.pipeline()
            p.xrevrange(self.tracked_key, count=1)
            msg = p.execute()

            index = None
            if msg and len(msg[0]) > 0:
                index = msg[0][0][0].decode("utf-8")

            if ((index is None) or
                (self.previous_index is not None and self.previous_index == index)):
                return None

            self.previous_index = index
            return msg[0][0][1]

        except Exception as e:
            logger.error(f"Error getting data: {e}")
            return None

    def run(self, conn):
        while True:
            data = self.get_data(conn)
            if data:
                result = self.update(data)
                # logger.debug(result)
                conn.xadd(self.prowled_key, result, maxlen=STREAMMAXLEN)