# YOLOv8-PyTorch

A PyTorch-based implementation of **YOLOv8**, built upon the **Ultralytics** framework.

## Requirements

- Windows 10  
- Python ≥ 3.9  
- Docker Desktop  

## Dataset

The dataset used in this project is self-collected for fall detection tasks and is publicly available on Roboflow Universe:

👉 https://universe.roboflow.com/data-whh4s/falldetection2

## Overview

Task: Human fall detection

Annotations: Bounding boxes and keypoints (YOLO format)

Data splits: Train / Validation / Test

## Quality Check & Performance

The dataset was reviewed using Roboflow tools to ensure correct annotations, data diversity, and overall quality.
Using this dataset, the YOLOv8 model achieved a mAP@0.5 (map50) of 75.8% on the validation set, with a real-time inference speed of 15–24 FPS, evaluated on an NVIDIA GeForce GTX 1660 Ti.

## Running the Code

### 1. Configure the RTSP Stream

Before executing the code, the RTSP stream of the camera must be identified and properly configured in the file:

```
rtsp_server/mediamtx.yml
```

Locate the following section:

```yaml
paths:
  cam1:
    source: rtsp://admin:SNXLBC@192.168.78.110:554/H.264 #75
    sourceProtocol: tcp
```

Replace the value of the `source` field with the RTSP URL of the target camera.  
Ensure that both the computer and the camera are connected to the same Wi-Fi network or local area network (LAN).

### 2. Start the Docker Services

Open **Command Prompt (cmd)** and run:

```bash
docker-compose up -d
```

### 3. Check the Visualization Service Logs

```bash
docker compose logs visualization_service
```

### 4. View the Results

Copy the provided local URL from the logs and open it in a modern web browser (e.g., Chrome, Edge) to view the results.
