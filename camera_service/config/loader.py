import yaml
import json

def load_env():
    with open('env.json') as stream:
        env_cfg = json.load(stream)
        return env_cfg['env']

env = load_env()

def load_config():
    file_name = f'config/config.{env}.yml'
    with open(file_name, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

cfg = load_config()

# CONFIGS
CAMERAS = cfg['cameras']
STREAM_MAXLEN = cfg['stream']['maxlen']
TARGET_FPS = cfg['stream']['target_fps']
REDIS_HOSTNAME = cfg['redis']['host_name']
REDIS_PORT = cfg['redis']['port']