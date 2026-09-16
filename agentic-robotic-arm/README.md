# SO-100 Robotic Arm AI Control System

An autonomous robotic manipulation system powered by a hybrid **LangGraph + smolagents** architecture, integrating local vision-language models (VLM), edge execution via FastAPI, and physical hardware control through the **LeRobot** framework.

---

## Repository Structure & File Breakdown

* **`agents/reasoner_agent.py`**: Implements the core `ReasoningAgent` class leveraging LangGraph and Ollama (`qwen2.5:3b`). It manages conversational intent routing, VLM workspace perception using camera capture tools, planning logic, server orchestration for robot control policies, and action verification loops.
* **`agents/action_agent.py`**: Built with FastAPI and `smolagents` (`Qwen/Qwen2.5-Coder-7B-Instruct`), acting as an edge execution node. It connects to physical hardware via LeRobot (`SO100FollowerConfig`) and runs HTTP endpoints (`/execute`) to trigger robot manipulation loops.
* **`agents/tools/cameras_tool.py`**: Contains the `@tool` decorator function `get_cameras_images`, which interfaces with local V4L2 USB cameras (Intel RealSense and Logitech Webcam) via OpenCV, compresses frames to JPEG, and routes visual analysis payloads to a cloud VLM endpoint.
* **`main.py`**: The entry point for running the interactive loop, initializing the `ReasoningAgent`, maintaining conversational state threads via LangGraph memory checkpointing, and routing user input to the graph workflow.
* **`pyproject.toml`**: Defines project configurations, Python version constraints (`>=3.13`), and core dependencies including FastAPI, LangChain, LeRobot, OpenCV, smolagents, and Uvicorn.
* **`uv.lock`**: Lockfile tracking exact package versions and dependency graphs resolved by UV.
* **`graph_architecture.png`**: Visual architectural diagram representing the compiled LangGraph workflow states and transitions.

---

## System Architecture & Workflow

1. **Intent Routing:** User inputs are processed by `initial_router` to determine whether the request is a general conversation (`assistant` node) or requires physical workspace interaction (`perception` node).
2. **Perception & VLM Verification:** The `perception` node calls `get_cameras_images` to capture real-time frames from workspace cameras, validated against structured JSON schemas (`WorkspaceStatus`) to check object locations (e.g., blue or red objects relative to a transparent box).
3. **Cloud Planning:** The `reasoner` node evaluates scene states against available low-level actions and produces structured plans (`PlannerOutput`) containing task statuses like `action_required`, `task_already_done`, or `information_only`.
4. **Edge Execution:** Approved action plans trigger the `action` node, sending requests to the FastAPI edge server (`agents/action_agent.py`), which starts the LeRobot policy server and controls the physical SO-100 robotic arm.
5. **Action Verification:** Post-execution, the `verify_action` node takes new camera snapshots to evaluate whether the physical workspace matches expected target states, continuing or ending the action queue based on success status.

---

## Model Training & Data Collection

To train policies (e.g., Diffusion Policy or ACT) for the robotic arm using the LeRobot framework, follow the data collection and training pipeline below:

### 1. Data Collection (Teleoperation)
Record demonstration episodes using the leader-follower setup or teleoperation interface:
```bash
python -m lerobot.scripts.control_robot \
    --robot-type so100 \
    --control-mode teleoperate \
    --fps 30
```
To save recorded demonstrations to a dataset repository:
```bash
python -m lerobot.scripts.control_robot \
    --robot-type so100 \
    --control-mode record \
    --repo-id "your-username/so100_task_name" \
    --num-episodes 50 \
    --fps 30
```

### 2. Dataset Visualization & Inspection
Inspect recorded episodes, video frames, and action states:
```bash
python -m lerobot.scripts.visualize_dataset \
    --repo-id "your-username/so100_task_name"
```

### 3. Policy Training Programmatic Execution
To launch training programmatically (e.g., using ACT policy configurations, asynchronous data loading, and custom worker parameters):
```python
import sys
import subprocess

cmd = [
    sys.executable, "-m", "lerobot.scripts.train",
    "policy=act",
    "env=aloha_sim_transfer_cube_human",
    "dataset_repo_id=lerobot/aloha_sim_transfer_cube_human",
    "batch_size=8",
    "num_workers=4",
    "training.eval_freq=5000",
    "training.save_freq=5000",
    "training.steps=100000",
    "device=cuda",
    "policy.chunk_size=100",
    "policy.n_action_steps=100",
]

subprocess.run(cmd, check=True)
```

Alternatively, run training directly via CLI using Hydra configurations:
```bash
python -m lerobot.scripts.train \
    policy=act \
    env=aloha_sim_transfer_cube_human \
    dataset_repo_id=lerobot/aloha_sim_transfer_cube_human \
    batch_size=8 \
    num_workers=4 \
    training.eval_freq=5000 \
    training.save_freq=5000 \
    training.steps=100000 \
    device=cuda \
    policy.chunk_size=100 \
    policy.n_action_steps=100
```

### 4. Policy Evaluation & Deployment
Evaluate the trained policy locally on the physical hardware:
```bash
python -m lerobot.scripts.control_robot \
    --robot-type so100 \
    --control-mode evaluate \
    --policy.path="outputs/train/act_aloha/checkpoints/last/pretrained_model"
```

---

## Installation & Setup

1. Ensure Python `>=3.13` and [uv](https://github.com/astral-sh/uv) are installed.
2. Install dependencies using uv:
   ```bash
   uv sync
   ```
3. Run the edge action server:
   ```bash
   python agents/action_agent.py
   ```
4. Start the interactive reasoning agent loop:
   ```bash
   python main.py
   ```

