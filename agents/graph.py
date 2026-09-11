from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
from inference_agent import run_inference
from guardrail_agent import check_guardrail
from drift_monitor_agent import check_drift


class AgentState(TypedDict):
    image_path: str
    prediction: Optional[dict]
    guardrail_result: Optional[dict]
    drift_result: Optional[dict]


def inference_node(state: AgentState) -> AgentState:
    prediction = run_inference(state["image_path"])
    return {**state, "prediction": prediction}


def guardrail_node(state: AgentState) -> AgentState:
    guardrail_result = check_guardrail(state["prediction"])
    return {**state, "guardrail_result": guardrail_result}


def drift_node(state: AgentState) -> AgentState:
    drift_result = check_drift()
    return {**state, "drift_result": drift_result}


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("inference", inference_node)
    graph.add_node("guardrail", guardrail_node)
    graph.add_node("drift_check", drift_node)

    graph.set_entry_point("inference")
    graph.add_edge("inference", "guardrail")
    graph.add_edge("guardrail", "drift_check")
    graph.add_edge("drift_check", END)

    return graph.compile()


if __name__ == "__main__":
    app = build_graph()
    final_state = app.invoke({"image_path": "data/processed/val/patches/patches_241.jpg"})
    print("\nFinal state:")
    print(final_state)