import sys
import signal
import multiprocessing
from multiprocessing import Process

import redis
from config.loader import REDIS_HOSTNAME, REDIS_PORT, CAMERAS
from fall_detect import Falling





def signal_handler(sig, frame):
    for process in multiprocessing.active_children():
        process.terminate()
    sys.exit(1)

signal.signal(signal.SIGINT, signal_handler)
r = redis.Redis(host=REDIS_HOSTNAME, port=REDIS_PORT)

def run_falling(camera_info):
    falling = Falling(camera_info["id"])
    falling.run(r)


if __name__ == "__main__":
    processes = list()

    for camera_info in CAMERAS:
        p = Process(target=run_falling, args=(camera_info,))
        p.start()
        processes.append(p)

    for process in processes:
        process.join()
