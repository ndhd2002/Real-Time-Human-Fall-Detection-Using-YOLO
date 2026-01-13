import sys
import signal
import multiprocessing
from multiprocessing import Process

import redis

from pose_estimate import PoseEstimator
from config.loader import REDIS_HOSTNAME, REDIS_PORT, CAMERAS


r = redis.Redis(host=REDIS_HOSTNAME, port=REDIS_PORT)

def run_tracker(camera_info):
    tracker = PoseEstimator(camera_info["id"])
    tracker.run(r)  

if __name__ == "__main__":
    processes = list()

    for camera_info in CAMERAS:
        print(camera_info)
        p = Process(target=run_tracker, args=(camera_info,))
        p.start()