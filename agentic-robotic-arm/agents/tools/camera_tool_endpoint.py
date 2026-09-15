from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from uvicorn.config import logger

from agents.tools.cameras_tool import get_cameras_images

app = FastAPI(title="PC Camera Perception API")


class PerceptionRequest(BaseModel):
    request_reason: str


@app.post("/analyze_workspace")
async def analyze_workspace(req: PerceptionRequest):
    """
    Endpoint that triggers the get_cameras_images tool.
    Accepts a reason (prompt), grabs the frames, sends them to the VLM, and returns the analysis.
    """
    logger.info(f"Received perception request: {req.request_reason}")

    try:
        result = get_cameras_images.invoke({"request_reason": req.request_reason})

        return {"status": "success", "analysis": result}
    except Exception as e:
        logger.error(f"Failed to run perception tool: {e}")
        return {"status": "error", "detail": str(e)}


if __name__ == "__main__":
    print("Starting PC Camera Perception API on http://0.0.0.0:8001...")
    uvicorn.run(app, host="0.0.0.0", port=8001)
