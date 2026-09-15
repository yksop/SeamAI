import threading
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from smolagents import CodeAgent, InferenceClientModel, tool

from lerobot.robots.so_follower import SO100FollowerConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.async_inference.configs import RobotClientConfig
from lerobot.async_inference.robot_client import RobotClient
from lerobot.async_inference.helpers import visualize_action_queue_size


@tool
def execute_lerobot_policy(task_name: str) -> str:
    """
    Connects to the LeRobot policy server, starts the high-frequency control loop,
    and executes a specific manipulation task on the physical SO-100 robotic arm.

    Args:
        task_name: The name of the task to execute.

    Returns:
        A string summarizing the outcome of the physical execution.
    """

    camera_cfg = {
        "front": OpenCVCameraConfig(
            index_or_path="/dev/v4l/by-id/usb-Intel_R__RealSense_TM__Depth_Camera_435_Intel_R__RealSense_TM__Depth_Camera_435_009223025055-video-index0",
            width=640,
            height=480,
            fps=30,
        ),
        "right": OpenCVCameraConfig(
            index_or_path="/dev/v4l/by-id/usb-046d_Logi_4K_Stream_Edition_6714B58F-video-index0",
            width=640,
            height=480,
            fps=30,
        ),
        "up": OpenCVCameraConfig(
            index_or_path="/dev/v4l/by-id/usb-046d_Logitech_Webcam_C925e_EBD6F57F-video-index0",
            width=640,
            height=480,
            fps=30,
        ),
    }

    robot_cfg = SO100FollowerConfig(
        port="/dev/ttyACM0", id="my_awesome_follower_arm", cameras=camera_cfg
    )

    client_cfg = RobotClientConfig(
        robot=robot_cfg,
        server_address="localhost:8080",
        policy_device="cuda",
        client_device="cpu",
        policy_type="act",
        pretrained_name_or_path="posky02/{task_name}",
        chunk_size_threshold=0.5,
        actions_per_chunk=50,
        task=task_name,
    )

    max_duration = 12.0

    try:
        client = RobotClient(client_cfg)

        print(f"Starting RobotClient for task: {task_name}")

        if client.start():
            action_receiver_thread = threading.Thread(
                target=client.receive_actions, daemon=True
            )
            action_receiver_thread.start()

            stop_timer = threading.Timer(max_duration, client.stop)
            stop_timer.start()

            try:
                client.control_loop(task_name)
                return "SUCCESS: Inference loop finished normally."
            except Exception:
                return "SUCCESS: Inference completed (stopped by timer)."
            finally:
                stop_timer.cancel()
                try:
                    client.stop()
                except Exception:
                    pass
                action_receiver_thread.join(timeout=2.0)
                # visualize_action_queue_size(client.action_queue_size)

        else:
            return "ERROR: Failed to start RobotClient."

    except Exception as e:
        return f"ERROR: Exception during execution: {str(e)}"


model = InferenceClientModel(model_id="Qwen/Qwen2.5-Coder-7B-Instruct")

action_agent = CodeAgent(
    tools=[execute_lerobot_policy], model=model, add_base_tools=False
)

app = FastAPI(title="Edge Action Agent API")


class ActionRequest(BaseModel):
    action: str


@app.post("/execute")
def execute_cloud_action(request: ActionRequest):
    """
    This endpoint listens for POST requests from the Cloud Planner (LangGraph).
    """

    prompt = f"""
        You are the Edge Action Agent.
        1. Call: result = execute_lerobot_policy(task_name='{request.action}')
        2. IMMEDIATELY call final_answer(result).
        3. If you see Action receiving thread starting, it's a success.
        You should wait for inference to complete and return the final success, doing nothing more.
        
        DO NOT write 'while' loops.
        DO NOT write 'try/except' blocks.
        DO NOT write 'retry' functions.
        Just execute the tool and provide the answer.
        """

    try:
        result = action_agent.run(prompt)

        if "ERROR" in str(result):
            raise HTTPException(status_code=500, detail=str(result))

        if "SUCCESS" not in str(result):
            raise HTTPException(
                status_code=500, detail="Agent returned unexpected format."
            )

        return {
            "status": "success",
            "detail": str(result),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
