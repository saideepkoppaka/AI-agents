import json
import sys
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from config import ENVIRONMENT
from agent_nodes import (
    fetch_code_from_gitlab,
    extract_information,
    synthesize_documentation,
    publish_to_confluence,
    check_completeness,
    flag_missing_information
)

# Define the state for our agent graph
class AgentState(TypedDict):
    model_path: str
    file_paths: List[str]
    file_contents: List[str]
    extracted_data: dict
    documentation: str
    missing_info: List[str]

def run_agent(model_path: str):
    """
    Initializes and runs the technical writer agent graph based on the environment.
    """
    # Define the workflow graph
    workflow = StateGraph(AgentState)

    # Add nodes to the graph
    workflow.add_node("fetch_files", fetch_code_from_gitlab)
    workflow.add_node("extract_info", extract_information)
    workflow.add_node("synthesize_doc", synthesize_documentation)
    workflow.add_node("publish_doc", publish_to_confluence)
    workflow.add_node("flag_missing", flag_missing_information)

    # Define the core edges
    workflow.set_entry_point("fetch_files")
    workflow.add_edge("fetch_files", "extract_info")
    workflow.add_conditional_edges(
        "extract_info",
        check_completeness,
        {"complete": "synthesize_doc", "incomplete": "flag_missing"}
    )
    workflow.add_edge("flag_missing", END)

    # --- Conditional Workflow ---
    if ENVIRONMENT == "production":
        print("--- RUNNING IN PRODUCTION MODE (FULLY AUTOMATED) ---")
        workflow.add_edge("synthesize_doc", "publish_doc")
        workflow.add_edge("publish_doc", END)
    else:
        print("--- RUNNING IN DEVELOPMENT MODE (HUMAN-IN-THE-LOOP) ---")
        workflow.add_edge("synthesize_doc", END) # Stop before publishing

    # Compile the graph
    app = workflow.compile()

    # Run the agent
    initial_state = {"model_path": model_path}
    final_state = app.invoke(initial_state)

    # --- Post-run actions based on environment ---
    if final_state.get("documentation"):
        if ENVIRONMENT == "development":
            with open("preview_session.json", "w") as f:
                json.dump(final_state, f, indent=4)
            print("\n--- GENERATION COMPLETE ---")
            print("Agent state saved to preview_session.json.")
            print("To review and publish, run: streamlit run preview_app.py")
        else:
            print("\n--- PRODUCTION RUN COMPLETE ---")
            print("Documentation has been published automatically.")
    else:
        print("\n--- AGENT RUN FAILED ---")
        print("Documentation was not generated due to missing information.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        model_directory_path = sys.argv[1]
    else:
        # Fallback for demonstration purposes
        model_directory_path = "models/your_new_model_directory"
        print(f"Usage: python main.py <path_to_model_directory>")
        print(f"Using default path: {model_directory_path}\n")

    print(f"Starting AI Technical Writer agent for: {model_directory_path}")
    run_agent(model_directory_path)
