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
REDIS_HOSTNAME = cfg["redis"]["HOSTNAME"]
REDIS_PORT = cfg["redis"]["PORT"]
STREAMMAXLEN = cfg["stream"]["STREAMMAXLEN"]
CAMERAS = cfg["cameras"]

THICKNESS = cfg["draw_params"]["THICKNESS"]
TEXTTHICKNESS = cfg["draw_params"]["TEXTTHICKNESS"]
TEXTSCALE = cfg["draw_params"]["TEXTSCALE"]
TRIANGLESIZE = cfg["draw_params"]["TRIANGLESIZE"]
OFFSET = cfg["draw_params"]["OFFSET"]
