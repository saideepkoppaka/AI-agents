# prompts.py

# This prompt is engineered to extract structured data and handle missing information.
EXTRACTION_PROMPT_TEMPLATE = """
You are an expert technical writer AI. Your task is to analyze the content of several code and configuration files for a new machine learning model pipeline and extract specific pieces of information.

Here are the files and their contents:
---
File: {file_paths[0]} (Mdl.yml)
Content:
{file_contents[0]}
---
File: {file_paths[1]} (Pipelines_model.yml)
Content:
{file_contents[1]}
---
File: {file_paths[2]} (Base_data.py)
Content:
{file_contents[2]}
---
File: {file_paths[3]} (Features.py)
Content:
{file_contents[3]}
---
File: {file_paths[4]} (Inference.py)
Content:
{file_contents[4]}
---

Based on the provided files, extract the following information. **If you cannot find a piece of information, you MUST return `null` for that field.** Do not guess or make up information.

1.  **pipeline_name**: Find the name of the pipeline. This is often defined in `Pipelines_model.yml`.
2.  **model_list**: Identify the model name(s) being used. Look for this in `Mdl.yml` under a 'name' or 'model_name' key.
3.  **execution_cadence**: Determine how often the pipeline runs (e.g., weekly, daily). This is usually found in `Pipelines_model.yml` as a cron expression or a schedule key.
4.  **output_description**: Describe the output of the pipeline. Look for comments in `Inference.py` or `Base_data.py` that describe the final DataFrame or output table.
5.  **feature_columns**: List the feature columns used by the model. The contents of `Features.py` should contain this list directly.
6.  **output_consumers**: Infer where the output data is used. Look for code in `Inference.py` that writes to a specific table, API, or location. If not explicitly stated, return null.

Return your findings as a single, well-formed JSON object.
"""

# This prompt synthesizes the extracted data into Confluence-friendly markdown.
# It's enhanced to create a much richer, more readable page.
SYNTHESIS_PROMPT_TEMPLATE = """
You are an expert technical writer AI creating a documentation page for Atlassian Confluence.
You have been given a JSON object with details about a machine learning pipeline.
Your task is to convert this information into a well-structured and visually appealing document using Confluence's markdown-style formatting.

**Formatting Instructions:**
- Use headings for each section.
- Create a main table for the key pipeline details.
- Use Confluence's info `{info}` and note `{note}` panels for descriptions and important callouts.
- Format the feature list inside a code block `{code}` for readability.
- Ensure the final output is clean, professional, and easy to read.

**JSON Data:**
```json
{extracted_data}
