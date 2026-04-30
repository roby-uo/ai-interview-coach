import logging
from typing import Literal
from langgraph.graph import StateGraph, END

from domain.schemas import InterviewState
from core.nodes import route_node, update_history_node, report_node
from core.skills import SKILL_NODE_MAP

logger = logging.getLogger(__name__)


def route_decision(state: InterviewState) -> str:
    if state.get("is_generating_report"):
        return "report"
    
    decision = state.get("decision", {})
    target_skill = decision.get("target_skill", "smart_redirection")
    
    skill_to_node = {
        "socratic_probe": "socratic",
        "scaffold_rescue": "scaffold",
        "judge_strike": "judge",
        "smart_redirection": "redirection",
        "teaching_supplement": "teaching"
    }
    return skill_to_node.get(target_skill, "redirection")


def build_interview_graph():
    graph = StateGraph(InterviewState)

    graph.add_node("route", route_node)
    graph.add_node("socratic", SKILL_NODE_MAP["socratic_probe"])
    graph.add_node("scaffold", SKILL_NODE_MAP["scaffold_rescue"])
    graph.add_node("judge", SKILL_NODE_MAP["judge_strike"])
    graph.add_node("redirection", SKILL_NODE_MAP["smart_redirection"])
    graph.add_node("teaching", SKILL_NODE_MAP["teaching_supplement"])
    graph.add_node("update_history", update_history_node)
    graph.add_node("report", report_node)

    graph.set_entry_point("route")

    graph.add_conditional_edges(
        "route",
        route_decision,
        {
            "report": "report",
            "socratic": "socratic",
            "scaffold": "scaffold",
            "judge": "judge",
            "redirection": "redirection",
            "teaching": "teaching"
        }
    )

    for skill_name in ["socratic", "scaffold", "judge", "redirection", "teaching"]:
        graph.add_edge(skill_name, "update_history")

    graph.add_edge("update_history", END)
    graph.add_edge("report", END)

    return graph
