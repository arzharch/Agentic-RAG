
import os
import json
# Updated import to address deprecation warning
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent

# Local imports
from src.prompts import (
    SUPERVISOR_PROMPT, FINANCIAL_AGENT_PROMPT, TECHNICAL_AGENT_PROMPT, 
    RISK_AGENT_PROMPT, PROJECT_MANAGEMENT_AGENT_PROMPT, FEEDBACK_AGENT_PROMPT, 
    PERFORMANCE_AGENT_PROMPT, GENERAL_AGENT_PROMPT, GRADER_PROMPT # Added missing import
)
from src.tools import all_tools

# --- NODE IMPLEMENTATIONS ---

def knowledge_base_node(state):
    """Creates a 'fast' knowledge base by reading the first 3 lines of each file."""
    if "knowledge_base" not in state or not state["knowledge_base"]:
        print("--- Creating Fast Knowledge Base ---")
        file_path = os.path.join(os.getcwd(), "files_to_work_with")
        if not os.path.exists(file_path):
            return {"knowledge_base": {}}

        files = [f for f in os.listdir(file_path) if f.endswith('.txt')]
        knowledge_base = {}
        for file in files:
            full_path = os.path.join(file_path, file)
            with open(full_path, 'r') as f:
                preview = "".join(f.readlines()[:3])
                knowledge_base[file] = preview.strip()
        print("--- Knowledge Base Created ---\n")
        return {"knowledge_base": knowledge_base}
    return {}

def supervisor_node(state):
    """The supervisor node that decides which agent to run next."""
    llm = ChatOllama(model="mistral")
    prompt = ChatPromptTemplate.from_template(SUPERVISOR_PROMPT)
    chain = prompt | llm | StrOutputParser()
    
    kb_string = json.dumps(state['knowledge_base'], indent=2)
    agent_to_run = chain.invoke({"query": state['original_query'], "knowledge_base": kb_string})
    agent_to_run = agent_to_run.strip().replace("'", '').replace('"', '').replace('`', '')
    return {"agent_to_run": agent_to_run}

def create_agent_node(system_prompt: str):
    """Factory function to create a self-correcting ReAct agent node."""
    llm = ChatOllama(model="mistral")
    
    # 1. The Agent that finds information and provides a preliminary answer
    base_prompt = hub.pull("hwchase17/react")
    prompt = ChatPromptTemplate.from_template(system_prompt + "\n\n" + base_prompt.template)
    agent = create_react_agent(llm, all_tools, prompt)
    executor = AgentExecutor(agent=agent, tools=all_tools, verbose=True, handle_parsing_errors=True, return_intermediate_steps=True)

    # 2. The Grader that validates the answer against the context
    grader_prompt = ChatPromptTemplate.from_messages([
        ("system", GRADER_PROMPT),
        ("human", "User Query: {query}\n\nContext:\n{context}\n\nProposed Answer:\n{answer}")
    ])
    grader_chain = grader_prompt | llm | StrOutputParser()

    def agent_node(state):
        """The node that runs the agent, grades the output, and retries if necessary."""
        max_attempts = 5
        attempts = 0
        
        while attempts < max_attempts:
            print(f"--- Agent Attempt {attempts + 1}/{max_attempts} ---")
            response = executor.invoke({
                "input": state['original_query']
            })

            # Extract the answer and the context (tool outputs)
            preliminary_answer = response['output']
            intermediate_steps = response.get('intermediate_steps', [])
            context = "\n".join([str(step[1]) for step in intermediate_steps])

            if not context:
                # If the agent failed to find any context, we can't validate.
                print("Agent failed to find any information. Retrying...")
                attempts += 1
                continue

            # Grade the answer
            grade = grader_chain.invoke({
                "query": state['original_query'],
                "context": context,
                "answer": preliminary_answer
            })

            if grade.lower().strip() == 'yes':
                print("--- Grade: VALID ---")
                return {"results": {state['agent_to_run']: preliminary_answer}}
            else:
                print(f"--- Grade: INVALID, Retrying... ---")
                attempts += 1
        
        # If loop finishes, it failed all attempts
        final_answer = "Could not find a confident answer after 5 attempts."
        return {"results": {state['agent_to_run']: final_answer}}
            
    return agent_node

def summarize_results_node(state):
    """Summarizes the results from the agent runs into a final answer."""
    print("--- Summarizing Results ---")
    llm = ChatOllama(model="mistral")
    prompt = ChatPromptTemplate.from_template("You are a summarizer. Your job is to take the results from other agents and formulate a single, cohesive, and easy-to-read final answer to the user's original query.\n\nUser's Query: {query}\n\nAgent Results:\n{results}")
    chain = prompt | llm | StrOutputParser()
    
    results_string = json.dumps(state['results'], indent=2)
    final_summary = chain.invoke({"query": state['original_query'], "results": results_string})
    
    return {"final_summary": final_summary}


# --- CREATE ALL NODES ---
financial_agent = create_agent_node(FINANCIAL_AGENT_PROMPT)
technical_agent = create_agent_node(TECHNICAL_AGENT_PROMPT)
risk_agent = create_agent_node(RISK_AGENT_PROMPT)
project_management_agent = create_agent_node(PROJECT_MANAGEMENT_AGENT_PROMPT)
feedback_agent = create_agent_node(FEEDBACK_AGENT_PROMPT)
performance_agent = create_agent_node(PERFORMANCE_AGENT_PROMPT)
general_agent = create_agent_node(GENERAL_AGENT_PROMPT)
