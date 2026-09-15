import os
import base64
import cv2
import logging
from typing import Any, Dict
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)


# Note: No @tool decorator here since main.py calls it normally
def get_cameras_images(request_reason: str) -> str:
    """Capture frames from the workspace cameras and get a visual analysis from the cloud VLM."""
    logger.info("Tool call: get_cameras_images")

    camera_cfgs = {
        "Intel(R) RealSense(TM) Depth Camera": "/dev/v4l/by-id/usb-Intel_R__RealSense_TM__Depth_Camera_435_Intel_R__RealSense_TM__Depth_Camera_435_009223025055-video-index0",
        "Logi 4K Stream Edition": "/dev/v4l/by-id/usb-046d_Logi_4K_Stream_Edition_6714B58F-video-index0",
        "Logitech Webcam C925e": "/dev/v4l/by-id/usb-046d_Logitech_Webcam_C925e_EBD6F57F-video-index0",
    }

    llm = ChatOpenAI(
        model="llama4:16x17b",
        base_url="http://edgewise.icedc.se/v1",
        api_key="2vp8crHoXKcKMxdM28bOITbgQ3euIptX",
    )

    message_content = [{"type": "text", "text": request_reason}]

    for cam_name, cam_path in camera_cfgs.items():
        # Resolve symlink to the actual hardware node (e.g., /dev/video8) for OpenCV
        real_path = os.path.realpath(cam_path)

        # Extract just the integer (e.g. 8 from /dev/video8) to bypass the OpenCV string warning entirely
        try:
            cam_index = int(real_path.replace("/dev/video", ""))
        except ValueError:
            cam_index = real_path  # Fallback if something weird happens

        logger.info(
            f"Opening local frame stream for: {cam_name} (Node index {cam_index})"
        )

        cap = cv2.VideoCapture(cam_index, cv2.CAP_V4L2)

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        if "Intel" in cam_name:
            for _ in range(60):
                cap.read()

        if not cap.isOpened():
            # Correctly appending to the list!
            message_content.append(
                {
                    "type": "text",
                    "text": f"{cam_name}: ERROR: Cannot open camera hardware pathway at index {cam_index}.",
                }
            )
            continue

        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            message_content.append(
                {
                    "type": "text",
                    "text": f"{cam_name}: ERROR: Camera returned an empty hardware frame.",
                }
            )
            continue

        success, buffer = cv2.imencode(".jpg", frame)
        if not success:
            message_content.append(
                {
                    "type": "text",
                    "text": f"{cam_name}: ERROR: Failed to compress frame to JPEG buffer.",
                }
            )
            continue

        image_bytes = buffer.tobytes()
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:image/jpeg;base64,{base64_image}"

        message_content.append({"type": "image_url", "image_url": {"url": data_url}})

    # VLM call is OUTSIDE the loop so it gathers all images first!
    try:
        message = HumanMessage(content=message_content)
        response = llm.invoke([message])
        return response.content

    except Exception as e:
        logger.error(f"Cloud VLM API Routing Error: {e}")
        return f"ERROR: VLM verification rejected connection: {e}"
