import streamlit as st
import json
import os
from agent_nodes import publish_to_confluence

st.set_page_config(page_title="Rxp AI Writer Preview", page_icon="🤖", layout="wide")

st.title("🤖 Rxp AI Technical Writer Preview & Approval")
st.markdown("Use this interface to review the AI-generated documentation before publishing it to Confluence.")

PREVIEW_FILE = "preview_session.json"

if not os.path.exists(PREVIEW_FILE):
    st.info(
        f"Waiting for a new preview session. "
        "Run `python main.py <path_to_model>` to generate documentation."
    )
    st.stop()

try:
    with open(PREVIEW_FILE, "r") as f:
        agent_state = json.load(f)
except Exception as e:
    st.error(f"Could not read or parse the preview file. Error: {e}")
    st.stop()

if agent_state and agent_state.get("documentation"):
    pipeline_name = agent_state.get("extracted_data", {}).get("pipeline_name", "Unknown Pipeline")

    st.header(f"Preview for: `{pipeline_name}`")
    
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Extracted Data (JSON)")
        st.json(agent_state.get("extracted_data", {}))

    with col2:
        st.subheader("Generated Document (Confluence Markdown)")
        st.markdown(agent_state.get("documentation", "No documentation was generated."))

    st.divider()

    st.header("Approval")
    st.warning("Please review the document carefully. Publishing is final.")

    if st.button("✅ Approve & Publish to Confluence", type="primary"):
        with st.spinner("Publishing to Confluence..."):
            try:
                publish_to_confluence(agent_state)
                st.success(f"Successfully published documentation for '{pipeline_name}'!")
                st.balloons()
                os.remove(PREVIEW_FILE)
                st.info("Preview file removed. Refresh the page to check for new sessions.")
            except Exception as e:
                st.error(f"An error occurred while publishing: {e}")
else:
    st.warning("The preview file seems to be empty or invalid.")
