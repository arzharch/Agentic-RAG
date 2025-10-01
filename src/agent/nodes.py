"""
Node implementations for the 4-node LangGraph workflow.
Each node performs a specific step in the agentic RAG pipeline.
"""

import json
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub

from config import LLM_MODEL, LLM_TEMPERATURE, MAX_FILES_TO_EXAMINE, MAX_REACT_ITERATIONS
from prompts.prompts import (
    QUERY_ANALYZER_PROMPT,
    EVIDENCE_GATHERER_PROMPT, SYNTHESIS_ANALYZER_PROMPT
)
from tools.tools import all_tools
from retrieval.vectorstore import vectorstore_manager


def query_analyzer_node(state):
    """
    NODE 1: QUERY ANALYZER
    
    Analyzes the user's query to understand what information is needed.
    Uses LLM reasoning to break down complex queries into structured components.
    
    Args:
        state: GraphState containing original_query
        
    Returns:
        dict: Updated state with query_analysis
    """
    print("\n" + "="*60)
    print("NODE 1: QUERY ANALYZER")
    print("="*60)
    
    try:
        llm = ChatOllama(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
        parser = JsonOutputParser()
        prompt = ChatPromptTemplate.from_template(QUERY_ANALYZER_PROMPT)
        chain = prompt | llm | parser
        
        print(f"Analyzing query: {state['original_query']}")
        
        # Get structured analysis from LLM
        query_analysis = chain.invoke({"query": state['original_query']})
        
        print("\nQuery Analysis:")
        print(json.dumps(query_analysis, indent=2))
        print("="*60 + "\n")
        
        return {"query_analysis": query_analysis}
    
    except Exception as e:
        print(f"Error in query_analyzer_node: {e}")
        return {"error": f"Query analysis failed: {str(e)}"}


def retrieval_planner_node(state):
    """
    NODE 2: RETRIEVAL PLANNER (Rule-Based)
    
    Determines which files are most relevant to the query using vector similarity.
    This is now a rule-based node that does not use an LLM.
    
    Args:
        state: GraphState containing original_query
        
    Returns:
        dict: Updated state with file_scores
    """
    print("\n" + "="*60)
    print("NODE 2: RETRIEVAL PLANNER")
    print("="*60)
    
    try:
        # Get file relevance scores from vector store
        print("Computing file relevance scores...")
        file_scores = vectorstore_manager.get_file_relevance_scores(state['original_query'])
        
        file_scores_str = "\n".join([
            f"- {filename}: {score:.4f}" 
            for filename, score in file_scores.items()
        ])
        
        print(f"\nFile Relevance Scores (lower is better):\n{file_scores_str}")
        print("="*60 + "\n")
        
        return {"file_scores": file_scores}
    
    except Exception as e:
        print(f"Error in retrieval_planner_node: {e}")
        return {"error": f"File relevance scoring failed: {str(e)}"}


def evidence_gatherer_node(state):
    """
    NODE 3: EVIDENCE GATHERER (ReAct Agent)
    
    Uses a ReAct agent to gather specific evidence from the selected files.
    The agent can use tools to search and read files.
    
    Args:
        state: GraphState containing original_query, query_analysis, and file_scores
        
    Returns:
        dict: Updated state with evidence
    """
    print("\n" + "="*60)
    print("NODE 3: EVIDENCE GATHERER (ReAct Agent)")
    print("="*60)
    
    try:
        llm = ChatOllama(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
        
        # The ReAct prompt has 'input', 'tools', 'tool_names', 'agent_scratchpad'
        react_prompt = hub.pull("hwchase17/react")
        
        # Prepend our custom instructions to the ReAct system message
        react_prompt.template = EVIDENCE_GATHERER_PROMPT + "\n\n" + react_prompt.template
        
        agent = create_react_agent(llm, all_tools, react_prompt)
        agent_executor = AgentExecutor(
            agent=agent,
            tools=all_tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=MAX_REACT_ITERATIONS,
            return_intermediate_steps=True
        )
        
        # Construct a detailed input string for the agent
        file_scores_str = "\n".join([
            f"- {filename}: {score:.4f}" 
            for filename, score in state['file_scores'].items()
        ])
        
        input_string = (
            f"Original Query: {state['original_query']}\n\n"
            f"Query Analysis:\n{json.dumps(state['query_analysis'], indent=2)}\n\n"
            f"Available Files with Relevance Scores (lower is better):\n{file_scores_str}"
        )
        
        print("\nAgent is being invoked with the following context:")
        print("--------------------------------------------------")
        print(input_string)
        print("--------------------------------------------------")
        
        # Run the agent
        response = agent_executor.invoke({"input": input_string})
        
        # Extract evidence from intermediate steps
        evidence = []
        for action, observation in response.get('intermediate_steps', []):
            evidence.append({
                'action': str(action.tool),
                'input': str(action.tool_input),
                'observation': str(observation)[:500]  # Truncate long observations
            })
        
        # Also include the final output
        evidence.append({
            'type': 'final_output',
            'content': response['output']
        })
        
        print("\n" + "="*60)
        print(f"Evidence gathered: {len(evidence)} pieces")
        print("="*60 + "\n")
        
        return {"evidence": evidence}
    
    except Exception as e:
        print(f"Error in evidence_gatherer_node: {e}")
        return {"error": f"Evidence gathering failed: {str(e)}"}


def synthesis_analyzer_node(state):
    """
    NODE 4: SYNTHESIS ANALYZER
    
    Synthesizes the gathered evidence into a final answer.
    Uses Chain-of-Thought reasoning to provide transparent analysis.
    
    Args:
        state: GraphState containing all previous information
        
    Returns:
        dict: Updated state with reasoning_trace and final_answer
    """
    print("\n" + "="*60)
    print("NODE 4: SYNTHESIS ANALYZER")
    print("="*60)
    
    try:
        llm = ChatOllama(model=LLM_MODEL, temperature=LLM_TEMPERATURE)
        prompt = ChatPromptTemplate.from_template(SYNTHESIS_ANALYZER_PROMPT)
        chain = prompt | llm | StrOutputParser()
        
        print("Synthesizing evidence into final answer...")
        
        # Format evidence for the LLM
        evidence_str = json.dumps(state['evidence'], indent=2)
        
        # Get the synthesis
        synthesis_output = chain.invoke({
            "query": state['original_query'],
            "query_analysis": json.dumps(state['query_analysis'], indent=2),
            "evidence": evidence_str
        })
        
        # Split into reasoning and answer
        if "**REASONING:**" in synthesis_output and "**ANSWER:**" in synthesis_output:
            parts = synthesis_output.split("**ANSWER:**")
            reasoning = parts[0].replace("**REASONING:**", "").strip()
            answer = parts[1].strip()
        else:
            reasoning = "Direct synthesis without explicit reasoning trace"
            answer = synthesis_output
        
        print("\n" + "="*60)
        print("SYNTHESIS COMPLETE")
        print("="*60 + "\n")
        
        return {
            "reasoning_trace": reasoning,
            "final_answer": answer
        }
    
    except Exception as e:
        print(f"Error in synthesis_analyzer_node: {e}")
        return {"error": f"Synthesis failed: {str(e)}"}