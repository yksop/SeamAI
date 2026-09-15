import json
import operator
import threading
from typing import Annotated, Literal
from langgraph.checkpoint.memory import MemorySaver
import requests


from pydantic import BaseModel, Field

from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.policy_server import serve

from agents.tools.cameras_tool import get_cameras_images
from langgraph.graph import END, START, StateGraph
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage
from langgraph.graph.message import add_messages


def replace_list(current: list[str], new: list[str]) -> list[str]:
    return new


class WorkspaceStatus(BaseModel):
    blue_object_location: Literal[
        "outside_transparent_box", "inside_transparent_box", "not_present"
    ] = Field(description="Location of the blue object.")
    red_object_location: Literal[
        "outside_transparent_box", "inside_transparent_box", "not_present"
    ] = Field(description="Location of the red object.")


class PlannerOutput(BaseModel):
    explanation: str = Field(
        description="The reasoning behind the plan, or the direct answer to the user's question about the scene."
    )
    actions: list[str] = Field(
        default_factory=list,
        description="The list of actions to execute. EMPTY if no action is needed or if answering a question.",
    )
    task_status: Literal[
        "action_required", "task_already_done", "impossible", "information_only"
    ] = Field(
        description="Classify the intent: 'action_required' to move objects, 'task_already_done' if goal met, 'impossible' if missing, 'information_only' if the user is just asking where things are."
    )


class AgentState(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages]
    blue_object_location: str = "unknown"
    red_object_location: str = "unknown"
    planner_decision: str = "unknown"
    action_plan: Annotated[list[str], replace_list] = Field(default_factory=list)
    user_request: str = ""
    is_executing_plan: bool = False
    last_action_success: str = "unknown"


class ReasoningAgent:
    def __init__(self):
        self.tools = [get_cameras_images]
        self.llm = ChatOllama(
            model="qwen2.5:3b",
            temperature=0,
            num_ctx=32768,
            timeout=60,
            enable_thinking=True,
        )
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.planner_llm = self.llm.with_structured_output(PlannerOutput)
        self.available_actions = {
            "pick_up_the_blue_cube": {
                "description": "ONLY grab the blue cube and hold it. Use this if the user says 'do not move it' or 'just grab it'.",
                "expected_state": "The blue object is currently held by the robotic arm.",
            },
            "rotate_the_blue_cube": {
                "description": "Rotate the blue cube. Use this if the user says 'rotate the blue cube'.",
                "expected_state": "The blue object is currently rotated.",
            },
            "pick_up_the_red_cube": {
                "description": "ONLY grab the red cube and hold it. Use this if the user says 'do not move it' or 'just grab it'.",
                "expected_state": "The red object is currently held by the robotic arm.",
            },
            "move_and_drop_the_object": {
                "description": "Grab the object AND move it into the transparent box.",
                "expected_state": "The object is currently inside the transparent box.",
            },
        }
        self.system_message = SystemMessage(
            content="You are a helpful assistant. You can converse and answer questions normally. "
            "Only use your `get_cameras_images` tool if the user asks you to look at the room, "
            "verify what you see, or check physical objects."
        )

        self.graph = self.build_graph()
        try:
            png_bytes = self.graph.get_graph().draw_mermaid_png()
            with open("graph_architecture.png", "wb") as f:
                f.write(png_bytes)
        except Exception as e:
            print(f"Error generating graph visualization: {e}")

    def initial_router(self, state: AgentState) -> Literal["perception", "assistant"]:
        """Use the LLM to classify the user's intent. User could either be asking to execute an action (which requires
        perception) or just having a general conversation."""

        if state.is_executing_plan and len(state.action_plan) > 0:
            return "perception"

        last_msg = state.messages[-1].content

        router_prompt = SystemMessage(
            content="Analyze the user's message. "
            "If the user wants to DO an action (e.g., move an object) OR wants to LOOK at the scene (e.g., 'where is', 'what do you see', 'use camera'), respond with exactly: 'perception'. "
            "If it's just a general conversation (e.g., 'hello', 'how are you'), respond with exactly: 'assistant'."
        )

        try:
            classification = (
                self.llm.invoke([router_prompt, HumanMessage(content=last_msg)])
                .content.strip()
                .lower()
            )

            state.user_request = last_msg

            if "perception" in classification:
                return "perception"
            else:
                return "assistant"

        except Exception as e:
            print(f"Router LLM error: {e}")
            exit()

    def perception(self, state: AgentState):
        """Calls the tool to analyze the scene in the real world"""

        schema_str = WorkspaceStatus.model_json_schema()

        prompt_input = (
            "Look at the workspace. Find the blue object and the transparent box.\n"
            "You MUST return a JSON object exactly like this:\n"
            '{"blue_object_location": "insert_value_here", "red_object_location": "insert_value_here"}\n'
            "Do not include any other text, explanations, or markdown. ALl the information you need is in the JSON schema: \n"
            f"{schema_str}"
        )

        try:
            vlm_perception = get_cameras_images.invoke({"request_reason": prompt_input})

            data = json.loads(vlm_perception)
            validated_status = WorkspaceStatus(**data)
            b_loc = validated_status.blue_object_location
            r_loc = validated_status.red_object_location
        except Exception as e:
            vlm_perception = f"Camera Error: {e}"
            b_loc = "unknown"
            r_loc = "unknown"

        return {
            "blue_object_location": b_loc,
            "red_object_location": r_loc,
            # "messages": [SystemMessage(content=vlm_perception)],
        }

    def reasoner(self, state: AgentState):
        blue_loc = state.blue_object_location
        red_loc = state.red_object_location
        usr_req = state.messages[-1].content if state.messages else "No request"

        actions_formatted = "\n".join(
            [
                f"- '{name}': {data['description']}"
                for name, data in self.available_actions.items()
            ]
        )

        reasoner_prompt = SystemMessage(
            content=f"""You are a Robotic Planner. Output valid JSON matching the user's intent.

        CURRENT WORLD STATE: 
        - Blue Object: {blue_loc}
        - Red Object: {red_loc}

        USER REQUEST: "{usr_req}"

        AVAILABLE ACTIONS TO CHOOSE FROM:
        {actions_formatted}

        EXAMPLES OF CORRECT BEHAVIOR:

        Scenario 1: User wants to grab the object, and it is outside.
        User Request: "pick up the blue object" OR "just grab it, don't put it in the box"
        World State: Blue Object is outside_transparent_box
        JSON Output: {{"explanation": "Grabbing the blue object.", "task_status": "action_required", "actions": ["pick up the blue object"]}}

        Scenario 2: User wants to move the object, and it is outside.
        User Request: "move the blue object into the box"
        World State: Blue Object is outside_transparent_box
        JSON Output: {{"explanation": "Moving the blue object into the transparent box.", "task_status": "action_required", "actions": ["move and drop the object"]}}

        Scenario 3: User wants to move the object, but it is ALREADY inside.
        User Request: "move the blue object into the box"
        World State: Blue Object is inside_transparent_box
        JSON Output: {{"explanation": "The object is already inside the transparent box.", "task_status": "task_already_done", "actions": []}}

        Now, generate the JSON for the CURRENT WORLD STATE and USER REQUEST."""
        )

        try:
            plan: PlannerOutput = self.planner_llm.invoke(
                [reasoner_prompt, HumanMessage(content=usr_req)]
            )
            if plan.task_status == "action_required" and plan.actions:
                decision = "execute_action"
                if (
                    not hasattr(self, "server_thread")
                    or not self.server_thread.is_alive()
                ):
                    self.server_thread = threading.Thread(
                        target=self.start_lerobot_server, daemon=True
                    )
                    self.server_thread.start()
            elif plan.task_status == "task_already_done":
                decision = "task_done"
            else:
                decision = "impossible"

            print(
                f"Planner Output:\n- Explanation: {plan.explanation}\n- Task Status: {plan.task_status}\n- Actions: {plan.actions}"
            )

            return {
                "action_plan": plan.actions,
                "planner_decision": decision,
                "messages": [AIMessage(content=plan.explanation)],
            }

        except Exception as e:
            print(f"Planner LLM Error: {e}")
            return {"planner_decision": "impossible", "action_plan": []}

    def reasoner_router(self, state: AgentState) -> Literal["action", "__end__"]:
        """Routes based on the Reasoner's decision"""
        if state.planner_decision == "execute_action" and len(state.action_plan) > 0:
            return "action"
        else:
            return "__end__"

    def start_lerobot_server(self):
        config = PolicyServerConfig(
            host="localhost",
            port=8080,
        )
        serve(config)

    def action(self, state: AgentState):
        if not state.action_plan:
            return {}

        current_action = state.action_plan[0]

        try:
            response = requests.post(
                "http://localhost:8000/execute",
                json={"action": current_action},
                timeout=3600,
            )

            if response.status_code == 200:
                result_msg = response.json().get(
                    "detail", f"Executed: {current_action}"
                )

            else:
                result_msg = f"Failed to execute: {current_action}"

        except requests.exceptions.ConnectionError:
            result_msg = f"Connection failed for: {current_action}"

        return {
            "messages": [AIMessage(content=result_msg)],
        }

    def verify_action(self, state: AgentState):
        if not state.action_plan:
            return {}

        current_action = state.action_plan[0]

        action_data = self.available_actions.get(current_action, {})

        target_state = action_data.get(
            "expected_state", f"The action '{current_action}' was completed."
        )

        prompt_input = (
            f"Analyze the workspace. We just attempted an action. "
            f"If successful, the physical scene MUST look like this: '{target_state}'\n"
            "Does the current scene match this expected state?\n"
            "You MUST return a JSON object exactly like this:\n"
            '{"reasoning": "Describe exactly what you see in the images step-by-step", "success": "YES" or "NO"}'
        )

        try:
            vlm_perception = get_cameras_images.invoke({"request_reason": prompt_input})

            cleaned_output = (
                vlm_perception.replace("```json", "").replace("```", "").strip()
            )

            data = json.loads(cleaned_output)

            if isinstance(data, list) and len(data) > 0:
                data = data[0]

            success_status = data.get("success", "NO").upper()

        except json.JSONDecodeError:
            success_status = "NO"
        except Exception as e:
            print(f"Unexpected Verification Error: {e}")
            success_status = "NO"

        if success_status == "YES":
            return {
                "last_action_success": "YES",
                "action_plan": state.action_plan[1:],
            }
        else:
            return {"last_action_success": "NO"}

    def verify_router(self, state: AgentState) -> Literal["action", "__end__"]:
        if state.last_action_success == "YES":
            if len(state.action_plan) == 0:
                return "__end__"
            return "action"
        else:
            return "action"

    def assistant(self, state: AgentState):
        return {
            "messages": [
                self.llm_with_tools.invoke([self.system_message] + state.messages)
            ]
        }

    def build_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node("perception", self.perception)
        builder.add_node("assistant", self.assistant)
        builder.add_node("reasoner", self.reasoner)
        builder.add_node("action", self.action)
        builder.add_node("verify_action", self.verify_action)

        # 1. Start -> Router (Perception or Assistant)
        builder.add_conditional_edges(
            START,
            self.initial_router,
            {"perception": "perception", "assistant": "assistant"},
        )

        # 2. Perception -> Reasoner
        builder.add_edge("perception", "reasoner")

        # 3. Reasoner -> Action or End (This connects the Reasoner to the Action node)
        builder.add_conditional_edges(
            "reasoner",
            self.reasoner_router,
            {
                "action": "action",
                "__end__": END,
            },
        )

        builder.add_edge("action", "verify_action")

        builder.add_conditional_edges(
            "verify_action",
            self.verify_router,
            {
                "action": "action",
                "__end__": END,
            },
        )

        builder.add_edge("assistant", END)

        return builder.compile(checkpointer=MemorySaver())
