"""
GraphState definition for the LangGraph workflow.
Tracks the progression of information through the 4-node pipeline.
"""

from typing import List, Dict
from typing_extensions import TypedDict


class GraphState(TypedDict):
    """
    State object that flows through the LangGraph workflow.
    
    Attributes:
        original_query (str): The user's original question
        query_analysis (Dict): Structured breakdown from Node 1 (Query Analyzer)
            - information_needed: List of info types to find
            - time_periods: Any specific dates/periods mentioned
            - metrics_requested: Quantitative data needed
            - inference_required: Whether synthesis/inference is needed
        
        selected_files (List[str]): Files identified by Node 2 (Retrieval Planner)
        
        evidence (List[Dict]): Raw evidence gathered by Node 3 (Evidence Gatherer)
            - Each dict contains: {source, content, relevance}
        
        reasoning_trace (str): Chain-of-thought from Node 4 (Synthesis Analyzer)
        
        final_answer (str): The complete answer with citations
        
        error (str): Any errors encountered during processing
    """
    original_query: str
    query_analysis: Dict
    file_scores: Dict[str, float]
    evidence: List[Dict]
    reasoning_trace: str
    final_answer: str
    error: str