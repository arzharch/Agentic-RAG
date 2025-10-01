"""
Main entry point for the LangGraph Agentic RAG system.
Provides a command-line interface for querying documents.
"""

import sys
from agent.graph import graph
from retrieval.vectorstore import vectorstore_manager
from config import VERBOSE


def initialize_system():
    """
    Initialize the system by building the vector store.
    This only needs to be done once (or when documents change).
    """
    print("\n" + "="*60)
    print("INITIALIZING AGENTIC RAG SYSTEM")
    print("="*60)
    
    try:
        vectorstore_manager.build_vectorstore()
        print("✓ System initialized successfully\n")
        return True
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        return False


def run_query(query: str):
    """
    Run a query through the LangGraph workflow.
    
    Args:
        query (str): The user's question
    """
    print("\n" + "="*60)
    print(f"QUERY: {query}")
    print("="*60)
    
    # Initialize state
    initial_state = {
        "original_query": query,
        "query_analysis": {},
        "file_scores": {},
        "evidence": [],
        "reasoning_trace": "",
        "final_answer": "",
        "error": ""
    }
    
    try:
        # Run the graph
        final_state = None
        for output in graph.stream(initial_state):
            # Stream outputs to see progress
            if VERBOSE:
                for node_name, node_output in output.items():
                    if node_output.get("error"):
                        print(f"\n⚠️  Error in {node_name}: {node_output['error']}")
            final_state = output
        
        # Display results
        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        
        if final_state:
            # Get the last node's output
            last_output = list(final_state.values())[0]
            
            # Check for errors
            if last_output.get("error"):
                print(f"\n⚠️  Error: {last_output['error']}")
                return
            
            # Display reasoning trace if available
            if last_output.get("reasoning_trace"):
                print("\n📊 REASONING PROCESS:")
                print("-" * 60)
                print(last_output['reasoning_trace'])
                print()
            
            # Display final answer
            if last_output.get("final_answer"):
                print("\n💡 ANSWER:")
                print("-" * 60)
                print(last_output['final_answer'])
            else:
                print("\n⚠️  No answer was generated.")
        else:
            print("\n⚠️  Graph execution produced no output.")
        
        print("\n" + "="*60 + "\n")
    
    except Exception as e:
        print(f"\n⚠️  Error executing query: {e}")
        import traceback
        traceback.print_exc()


def main():
    """
    Main CLI loop.
    """
    # Initialize system
    if not initialize_system():
        print("Failed to initialize. Exiting.")
        sys.exit(1)
    
    print("="*60)
    print("AGENTIC RAG SYSTEM - READY")
    print("="*60)
    print("Ask questions about your documents.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    # Query loop
    while True:
        try:
            query = input("🔍 Your question: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("\nGoodbye! 👋\n")
                break
            
            run_query(query)
        
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye! 👋\n")
            break
        except Exception as e:
            print(f"\nUnexpected error: {e}\n")


if __name__ == "__main__":
    main()