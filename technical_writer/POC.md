# POC Plan & Testing Guide: AI Technical Writer

## 1. Introduction & Objectives

This document outlines the plan and testing procedures for the Proof of Concept (POC) of the Rxp AI Technical Writer.

The primary objective of this POC is to validate the feasibility of using a LangGraph-based AI agent (powered by GPT-4o) to automatically generate and maintain technical documentation for ML model pipelines by parsing source code and configuration files.

### Success Criteria:
- **C1**: The agent can successfully parse a target model's files from GitLab.
- **C2**: The agent accurately extracts all required metadata fields (pipeline name, models, cadence, etc.).
- **C3**: The agent correctly identifies and flags when required information is missing, without hallucinating.
- **C4**: The agent generates a well-formatted, human-readable Confluence page.
- **C5**: The Human-in-the-Loop (HITL) review process via the Streamlit app works as designed.

---

## 2. Architecture Overview

The POC consists of two main workflows controlled by an `ENVIRONMENT` flag:

1.  **Development (HITL Workflow)**:
    * `main.py` is triggered, which runs the agent to generate documentation.
    * The output is saved to a `preview_session.json` file.
    * A team member runs `preview_app.py` to launch a Streamlit application.
    * The documentation is reviewed and can be approved for publishing to Confluence.
2.  **Production (Automated Workflow)**:
    * `main.py` is triggered.
    * The agent generates the documentation and publishes it directly to Confluence without any human intervention.

---

## 3. Setup Instructions

1.  **Clone the Repository**:
    * Ensure you have the `technical-writer-agent/` directory on your local machine.

2.  **Configure Environment Variables**:
    * Create a file named `.env` in the root of the project.
    * Populate it with your credentials. For this POC, ensure the environment is set to `development`.
    ```env
    # Set to 'development' for testing the preview app
    ENVIRONMENT="development"
    
    # API Keys and URLs
    OPENAI_API_KEY="sk-..."
    GITLAB_PRIVATE_TOKEN="glpat-..."
    CONFLUENCE_URL="[https://your-domain.atlassian.net](https://your-domain.atlassian.net)"
    CONFLUENCE_USERNAME="your-email@example.com"
    CONFLUENCE_API_TOKEN="your-atlassian-api-token"
    ```

3.  **Update Configuration**:
    * Open `config.py` and verify that `GITLAB_PROJECT_ID`, `CONFLUENCE_SPACE_KEY`, and `CONFLUENCE_PARENT_PAGE_TITLE` point to valid targets in your GitLab and Confluence instances.

4.  **Install Dependencies**:
    * Open your terminal, navigate to the project directory, and run:
    ```bash
    pip install -r requirements.txt
    ```

---

## 4. How to Proceed with Testing

Follow these test cases to validate the agent's functionality. For each test, you will point the agent to a model directory within your `rxperso` repository.

### Test Case 1: The "Happy Path" - Full End-to-End Flow

**Objective**: Verify the agent works correctly when all information is present.

1.  **Preparation**:
    * Choose a model directory in your GitLab repository (e.g., `models/retention_model_v1`) that has all the required files (`Mdl.yml`, `Pipelines_model.yml`, etc.) and ensure all metadata is correctly filled in within those files.

2.  **Execution**:
    * Run the agent from your terminal:
        ```bash
        python main.py models/retention_model_v1
        ```
    * Launch the Streamlit preview app:
        ```bash
        streamlit run preview_app.py
        ```

3.  **Validation**:
    * In the Streamlit app, check the "Extracted Data (JSON)" panel. Does it accurately reflect the information from the source files? (Success Criterion **C2**)
    * Review the "Generated Document" panel. Is the formatting clean, with a table, info panels, and a code block? (Success Criterion **C4**)
    * Click the "✅ Approve & Publish to Confluence" button.
    * Navigate to your Confluence space. Verify that a new page has been created (or an existing one updated) with the correct content. (Success Criterion **C5**)

### Test Case 2: The "Missing Information" Path

**Objective**: Verify the agent's guardrail against hallucination.

1.  **Preparation**:
    * In your GitLab repository, navigate to a test model's `Pipelines_model.yml` file.
    * Temporarily comment out or delete the line that defines the execution cadence (e.g., the `schedule` key). Commit this change to the `dev` branch.

2.  **Execution**:
    * Run the agent, pointing it to this modified model directory:
        ```bash
        python main.py models/your_test_model
        ```

3.  **Validation**:
    * Observe the terminal output. The agent should print a message indicating that it failed and could not find the `execution_cadence`. (Success Criterion **C3**)
    * Verify that no `preview_session.json` file was created.
    * Verify that no page was created or updated on Confluence.

### Test Case 3: Direct-to-Production Workflow Simulation

**Objective**: Verify the fully automated workflow.

1.  **Preparation**:
    * In your `.env` file, change the environment setting:
        ```env
        ENVIRONMENT="production"
        ```
    * Choose a "happy path" model directory to test with.

2.  **Execution**:
    * Run the agent from your terminal:
        ```bash
        python main.py models/retention_model_v1
        ```

3.  **Validation**:
    * Observe the terminal output. It should indicate that it is running in "PRODUCTION MODE" and that it is publishing directly to Confluence.
    * Navigate to Confluence and verify that the page was created/updated immediately, without any manual steps.
    * Verify that no `preview_session.json` file was created, as the HITL step was skipped.
