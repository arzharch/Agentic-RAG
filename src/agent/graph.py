"""
LangGraph workflow assembly.
Connects the 4 nodes into a linear pipeline with proper state management.
"""

from langgraph.graph import StateGraph, END
from agent.state import GraphState
from agent.nodes import (
    query_analyzer_node,
    retrieval_planner_node,
    evidence_gatherer_node,
    synthesis_analyzer_node
)


def should_continue(state: GraphState) -> str:
    """
    Conditional edge function to handle errors.
    If any node sets an error, skip to END.
    Otherwise, continue to the next node.
    
    Args:
        state: Current GraphState
        
    Returns:
        str: "error" to end early, or "continue" to proceed
    """
    if state.get("error"):
        return "error"
    return "continue"


# ===== BUILD THE GRAPH =====

# Initialize the graph with our state schema
workflow = StateGraph(GraphState)

# Add all 4 nodes
workflow.add_node("query_analyzer", query_analyzer_node)
workflow.add_node("retrieval_planner", retrieval_planner_node)
workflow.add_node("evidence_gatherer", evidence_gatherer_node)
workflow.add_node("synthesis_analyzer", synthesis_analyzer_node)

# Set the entry point
workflow.set_entry_point("query_analyzer")

# Define the linear flow with error handling
# Node 1 -> Node 2
workflow.add_conditional_edges(
    "query_analyzer",
    should_continue,
    {
        "continue": "retrieval_planner",
        "error": END
    }
)

# Node 2 -> Node 3
workflow.add_conditional_edges(
    "retrieval_planner",
    should_continue,
    {
        "continue": "evidence_gatherer",
        "error": END
    }
)

# Node 3 -> Node 4
workflow.add_conditional_edges(
    "evidence_gatherer",
    should_continue,
    {
        "continue": "synthesis_analyzer",
        "error": END
    }
)

# Node 4 -> END
workflow.add_edge("synthesis_analyzer", END)

# Compile the graph
graph = workflow.compile()

# Export for langgraph dev
__all__ = ["graph"]