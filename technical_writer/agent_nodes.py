import gitlab
import json
from typing import Dict
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from atlassian import Confluence
from config import (
    GITLAB_URL, GITLAB_PRIVATE_TOKEN, GITLAB_PROJECT_ID,
    CONFLUENCE_URL, CONFLUENCE_USERNAME, CONFLUENCE_API_TOKEN,
    CONFLUENCE_SPACE_KEY, CONfluence_PARENT_PAGE_TITLE, MODEL_NAME, OPENAI_API_KEY
)
from prompts import EXTRACTION_PROMPT_TEMPLATE, SYNTHESIS_PROMPT_TEMPLATE

# --- GitLab Tool ---
gl = gitlab.Gitlab(GITLAB_URL, private_token=GITLAB_PRIVATE_TOKEN)
project = gl.projects.get(GITLAB_PROJECT_ID)

def fetch_code_from_gitlab(state: Dict) -> Dict:
    print("--- FETCHING FILES FROM GITLAB ---")
    model_path = state["model_path"]
    file_names = ["Mdl.yml", "Pipelines_model.yml", "Base_data.py", "Features.py", "Inference.py"]
    
    file_contents = []
    file_paths_for_prompt = []
    
    for file_name in file_names:
        full_path = f"{model_path}/{file_name}"
        try:
            file_content = project.files.get(file_path=full_path, ref='dev').decode().decode('utf-8')
            file_contents.append(file_content)
            file_paths_for_prompt.append(full_path)
        except gitlab.exceptions.GitlabError as e:
            print(f"Warning: Could not fetch file {full_path}. Error: {e}")
            file_contents.append("")
            file_paths_for_prompt.append(full_path)
            
    state["file_contents"] = file_contents
    state["file_paths"] = file_paths_for_prompt
    return state

# --- LLM-based Nodes ---
llm = ChatOpenAI(model=MODEL_NAME, api_key=OPENAI_API_KEY, temperature=0.0)

def extract_information(state: Dict) -> Dict:
    print("--- EXTRACTING INFORMATION USING LLM ---")
    prompt = PromptTemplate.from_template(EXTRACTION_PROMPT_TEMPLATE)
    extraction_chain = prompt | llm
    
    response = extraction_chain.invoke({
        "file_paths": state["file_paths"],
        "file_contents": state["file_contents"]
    })
    
    try:
        clean_response = response.content.strip().replace('```json', '').replace('```', '')
        extracted_data = json.loads(clean_response)
        state["extracted_data"] = extracted_data
        print("Successfully extracted data:", json.dumps(extracted_data, indent=2))
    except json.JSONDecodeError:
        print(f"Error: Failed to decode JSON from LLM response.")
        state["extracted_data"] = {}
    
    return state

def synthesize_documentation(state: Dict) -> Dict:
    print("--- SYNTHESIZING DOCUMENTATION ---")
    prompt = PromptTemplate.from_template(SYNTHESIS_PROMPT_TEMPLATE)
    synthesis_chain = prompt | llm
    
    doc_content = synthesis_chain.invoke({
        "extracted_data": json.dumps(state["extracted_data"], indent=2)
    }).content
    
    disclaimer = "\n\n---\n\n*Disclaimer: This page was generated automatically by the Rxp AI Technical Writer. Please verify all information for accuracy.*"
    state["documentation"] = doc_content + disclaimer
    return state

# --- Confluence Tool ---
confluence = Confluence(
    url=CONFLUENCE_URL,
    username=CONFLUENCE_USERNAME,
    password=CONFLUENCE_API_TOKEN,
    cloud=True
)

def publish_to_confluence(state: Dict) -> Dict:
    print("--- PUBLISHING TO CONFLUENCE ---")
    pipeline_name = state["extracted_data"].get("pipeline_name", "Unknown Pipeline")
    page_title = f"ML Pipeline: {pipeline_name}"
    
    page_id = confluence.get_page_id(space=CONFLUENCE_SPACE_KEY, title=page_title)
    
    if page_id:
        print(f"Page '{page_title}' already exists. Updating it.")
        confluence.update_page(
            page_id=page_id,
            title=page_title,
            body=state["documentation"],
            representation='wiki'
        )
    else:
        print(f"Creating new page '{page_title}'.")
        parent_page_id = confluence.get_page_id(CONFLUENCE_SPACE_KEY, CONFLUENCE_PARENT_PAGE_TITLE)
        if not parent_page_id:
             raise Exception(f"Parent page '{CONFLUENCE_PARENT_PAGE_TITLE}' not found.")

        confluence.create_page(
            space=CONFLUENCE_SPACE_KEY,
            title=page_title,
            body=state["documentation"],
            parent_id=parent_page_id,
            representation='wiki'
        )
        
    print(f"Successfully published documentation for '{page_title}' to Confluence.")
    return state

# --- Conditional Logic and Error Handling ---
def check_completeness(state: Dict) -> str:
    print("--- CHECKING DATA COMPLETENESS ---")
    extracted_data = state.get("extracted_data", {})
    missing_fields = [key for key, value in extracted_data.items() if value is None]
    
    if not extracted_data or missing_fields:
        state["missing_info"] = missing_fields
        print(f"Validation failed. Missing fields: {missing_fields}")
        return "incomplete"
    else:
        print("Validation successful. All data is present.")
        return "complete"

def flag_missing_information(state: Dict) -> Dict:
    print("--- ACTION: FLAGGING MISSING INFORMATION ---")
    missing = state.get('missing_info', ['Unknown'])
    print(f"CRITICAL: The agent could not find the following required fields: {missing}.")
    return state
