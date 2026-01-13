from typing import List, Optional, Union

import cv2
import numpy as np
from logs.log_handler import logger

from supervision.draw.color import Color, ColorPalette


class Annotator:
    def __init__(
        self,
        color: Union[Color, ColorPalette] = ColorPalette(10),
        thickness: int = 2,
        text_color: Color =  Color.from_hex("#000000"),
        text_scale: float = 0.5,
        text_thickness: int = 1,
        text_padding: int = 10,
        anchor = "center"
    ):
        self.color: Union[Color, ColorPalette] = color
        self.thickness: int = thickness
        self.text_color: Color = text_color
        self.text_scale: float = text_scale
        self.text_thickness: int = text_thickness
        self.text_padding: int = text_padding
        self.anchor = anchor

    def annotate(self, scene: np.ndarray, detections, fps: Optional[float] = None, skip_label= False):
        font = cv2.FONT_HERSHEY_SIMPLEX

        if fps is not None:
            fps_text = f"FPS: {int(fps)}"
            cv2.putText(
                scene, 
                fps_text, 
                (scene.shape[1] - 120, 30),
                font, 
                0.7, 
                (0, 255, 0), 
                2, 
                cv2.LINE_AA
            )

        for i in detections:
            x1, y1, x2, y2 = i["bbox"]
            class_id = i["id"] if i["id"] is not None else None

            if i["fall_detected"]:
                color = (0, 0, 255)
            else:
                color = (255, 0, 0) 

            cv2.rectangle(
                img=scene,
                pt1=(x1, y1),
                pt2=(x2, y2),
                color=color,
                thickness=self.thickness,
            )

            if self.anchor == "center":
                cv2.circle(
                    scene,
                    (int((x2 + x1)/2), y2),
                    radius=5,
                    color=(0, 255, 0),
                    thickness=-1,
                    lineType=cv2.LINE_AA,
                )
            elif self.anchor == "right":
                cv2.circle(
                    scene,
                    (x2, y2),
                    radius=5,
                    color=(0, 0, 255),
                    thickness=-1,
                    lineType=cv2.LINE_AA,
                )
            elif self.anchor == "left":
                cv2.circle(
                    scene,
                    (x1, y2),
                    radius=5,
                    color=(255, 0, 0),
                    thickness=-1,
                    lineType=cv2.LINE_AA,
                )

            keypoints = i["keypoints"]
            if keypoints and len(keypoints) == 8:
                for kp in keypoints:
                    x, y = int(kp[0]), int(kp[1])
                    cv2.circle(scene, (x, y), radius=3, color=(255, 165, 0), thickness=-1, lineType=cv2.LINE_AA)

            if i["fall_detected"]:
                object_status = "fall"
            else:
                object_status = "normal"

            if skip_label:
                continue

            text = f"ID: {i['id']} - Status: {object_status}"

            text_width, text_height = cv2.getTextSize(
                text=text,
                fontFace=font,
                fontScale=self.text_scale,
                thickness=self.text_thickness,
            )[0]

            text_x = x1 + self.text_padding
            text_y = y1 - self.text_padding

            text_background_x1 = x1
            text_background_y1 = y1 - 2 * self.text_padding - text_height

            text_background_x2 = x1 + 2 * self.text_padding + text_width
            text_background_y2 = y1

            cv2.rectangle(
                img=scene,
                pt1=(text_background_x1, text_background_y1),
                pt2=(text_background_x2, text_background_y2),
                color=(255, 0, 0),
                thickness=cv2.FILLED,
            )

            cv2.putText(
                img=scene,
                text=text,
                org=(text_x, text_y),
                fontFace=font,
                fontScale=self.text_scale,
                color=self.text_color.as_rgb(),
                thickness=self.text_thickness,
                lineType=cv2.LINE_AA,
            )

        return scene

