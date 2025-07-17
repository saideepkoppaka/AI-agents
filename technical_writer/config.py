import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Agent Configuration ---
# Controls the workflow: 'development' for human-in-the-loop, 'production' for full automation.
ENVIRONMENT = os.getenv("ENVIRONMENT", "development") 

# --- OpenAI Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = "gpt-4o"

# --- GitLab Configuration ---
GITLAB_URL = "https://gitlab.com"
GITLAB_PRIVATE_TOKEN = os.getenv("GITLAB_PRIVATE_TOKEN")
GITLAB_PROJECT_ID = "your_group/rxperso" # Example: "my-org/my-project"

# --- Confluence Configuration ---
CONFLUENCE_URL = os.getenv("CONFLUENCE_URL")
CONFLUENCE_USERNAME = os.getenv("CONFLUENCE_USERNAME")
CONFLUENCE_API_TOKEN = os.getenv("CONFLUENCE_API_TOKEN")
CONFLUENCE_SPACE_KEY = "MLOPS"
CONFLUENCE_PARENT_PAGE_TITLE = "MLOps Home"
