from fastapi import FastAPI
from langchain_core.messages import HumanMessage
from openai import BaseModel
import uvicorn
from agents.reasoner_agent import ReasoningAgent
from agents.tools.cameras_tool import get_cameras_images

if __name__ == "__main__":
    agent = ReasoningAgent()

    current_state = {"messages": []}

    app = FastAPI(title="PC Perception Server")

    class PerceptionRequest(BaseModel):
        request_reason: str

    @app.post("/analyze_workspace")
    async def analyze_workspace(req: PerceptionRequest):
        return get_cameras_images(req.request_reason)

    if __name__ == "__main__":
        uvicorn.run(app, host="0.0.0.0", port=8001)

    while True:
        try:
            user_input = ""

            while not user_input.strip():
                user_input = input("User: ")

            if user_input.lower() in ["exit", "quit"]:
                break

            current_state["messages"].append(HumanMessage(content=user_input))

            print("System: ", end="", flush=True)

            config = {"configurable": {"thread_id": "1"}}

            result = agent.graph.invoke(current_state, config)

            ai_message = result["messages"][-1]

            print(ai_message.content)

            current_state["messages"] = result["messages"]

            print()

        except KeyboardInterrupt:
            break
