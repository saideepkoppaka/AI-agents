# Comprehensive Evaluation Plan: AI Technical Writer Agent

## 1. Objective

The goal of this evaluation is to rigorously assess the AI technical writer agent's performance across three key dimensions: **Accuracy**, **Quality**, and **Efficiency**. This will provide the data needed to validate the POC, identify areas for improvement, and make an informed decision about deploying the agent to production.

## 2. Evaluation Dimensions & Metrics

We will use a "golden set" of 5-10 model directories from your repository. For each model in this set, a human expert (e.g., an ML engineer familiar with the model) will manually create the "ground truth" documentation. The agent's output will be compared against this ground truth.

### Dimension 1: Accuracy & Reliability (Quantitative)

This dimension measures how correctly the agent extracts and processes information.

| Metric                 | Description                                                                                             | How to Measure                                                                                                                             | Target |
| :--------------------- | :------------------------------------------------------------------------------------------------------ | :----------------------------------------------------------------------------------------------------------------------------------------- | :--- |
| **Extraction Accuracy** | The percentage of fields the agent extracts correctly from the source files.                          | For each field (pipeline_name, model_list, etc.), compare the agent's extracted JSON value to the ground truth. Score 1 for a match, 0 for a mismatch. | > 95%  |
| **Completeness Score** | The percentage of required fields the agent successfully finds.                                         | (Total Fields Found) / (Total Fields Required). This tests the agent's ability to locate all necessary information.                        | > 98%  |
| **Guardrail Effectiveness**| The agent's ability to correctly identify and flag missing information without hallucinating.           | Run the agent on a model where a key piece of information has been deliberately removed.                                                   | 100% (Must always flag, never invent) |
| **Factual Consistency** | Does the final synthesized document contradict the extracted JSON data?                                 | Manually review the generated Markdown against the extracted JSON. This is a check on the synthesis step.                                  | 100% (No contradictions) |

### Dimension 2: Quality of Output (Qualitative)

This dimension measures the human-perceived quality of the final documentation. This should be scored by 2-3 team members to get a balanced view.

| Metric                   | Description                                                                                             | How to Measure (Scale of 1-5)                                                              | Target          |
| :----------------------- | :------------------------------------------------------------------------------------------------------ | :----------------------------------------------------------------------------------------- | :-------------- |
| **Readability & Clarity** | Is the documentation easy to understand for a new team member? Is the language clear and concise?       | 1 = Confusing; 3 = Acceptable; 5 = Very clear and well-written.                            | Avg. Score > 4.0 |
| **Formatting Correctness** | Does the final page render correctly in Confluence? Are tables, code blocks, and info panels used appropriately? | 1 = Broken formatting; 3 = Functional but messy; 5 = Clean, professional formatting.       | Avg. Score > 4.5 |
| **Structural Coherence** | Does the document flow logically? Are the sections well-organized and easy to navigate?                 | 1 = Disorganized; 3 = Logically ordered; 5 = Exceptionally well-structured.                | Avg. Score > 4.0 |

### Dimension 3: Efficiency & Performance (Quantitative)

This dimension measures the operational cost and speed of the agent.

| Metric                 | Description                                                                                          | How to Measure                                                                                   | Target         |
| :--------------------- | :--------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------- | :------------- |
| **End-to-End Latency** | The total time taken from triggering the agent to the documentation being published (or saved for preview). | Time the execution of the `main.py` script.                                                      | < 60 seconds   |
| **Cost Per Run** | The estimated cost of the OpenAI API calls for a single agent run.                                   | Use the OpenAI API usage dashboard or calculate based on input/output tokens for the two LLM calls. | < $0.10 per run |

## 3. Evaluation Execution Plan

1.  **Phase 1: Preparation (The "Golden Set")**
    * **Action**: Select 5-10 representative model directories from your `rxperso` repository. Include a mix of simple and complex models.
    * **Action**: For each selected model, manually create the ideal Confluence documentation. This is your "ground truth."
    * **Action**: Create a spreadsheet to track the results for each metric across all test models.

2.  **Phase 2: Running the Tests**
    * **Action**: For each model in the golden set, run the agent using the `development` workflow (`python main.py <model_path>`).
    * **Action**: Before approving in Streamlit, record the **Extraction Accuracy** and **Completeness Score** by comparing the preview JSON to your ground truth.
    * **Action**: After approving, open the Confluence page and have your review team score the **Readability**, **Formatting**, and **Coherence**. Record the scores.
    * **Action**: Record the **Latency** (from the terminal) and **Cost** (from OpenAI dashboard) for each run.

3.  **Phase 3: Guardrail Test**
    * **Action**: Take one of the models from your golden set and create a temporary "broken" version where a key piece of information is missing.
    * **Action**: Run the agent on this broken model.
    * **Action**: Verify that the agent terminates and correctly reports the missing field. Record the result for **Guardrail Effectiveness**.

4.  **Phase 4: Analysis & Reporting**
    * **Action**: Aggregate the results in your spreadsheet. Calculate the average scores for all metrics.
    * **Action**: Create a summary report that compares the agent's performance against the target for each metric.
    * **Action**: Include qualitative feedback from the review team and any notable failure cases. This report will be the primary artifact for your POC presentation to leadership.

By following this detailed evaluation plan, you will gain deep insights into the agent's capabilities and be well-prepared to make data-driven decisions for the future of this project.
