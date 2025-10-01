import sys
import os
import datetime
import re
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
            
            reasoning = last_output.get("reasoning_trace", "No reasoning trace available.")
            answer = last_output.get("final_answer", "No answer was generated.")

            # Display reasoning trace if available
            if reasoning:
                print("\n📊 REASONING PROCESS:")
                print("-" * 60)
                print(reasoning)
                print()
            
            # Display final answer
            if answer:
                print("\n💡 ANSWER:")
                print("-" * 60)
                print(answer)
            
            # --- Save the report ---
            try:
                reports_dir = os.path.join(os.getcwd(), "reports")
                os.makedirs(reports_dir, exist_ok=True)

                slug = re.sub(r'[^\w-]', ' ', query).strip().lower()
                slug = re.sub(r'[\s-]+', '-', slug)[:50]
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{timestamp}_{slug}.txt"
                filepath = os.path.join(reports_dir, filename)

                report_content = (
                    f"QUERY:\n{query}\n\n"
                    f"============================================================\n"
                    f"REASONING PROCESS:\n============================================================\n"
                    f"{reasoning}\n\n"
                    f"============================================================\n"
                    f"FINAL ANSWER:\n============================================================\n"
                    f"{answer}"
                )

                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(report_content)

                print(f"\n\n✅ Report saved to: {filepath}")

            except Exception as e:
                print(f"\n\n⚠️  Failed to save report: {e}")

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
