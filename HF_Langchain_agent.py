import sqlite3
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_core.language_models.llms import LLM
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.tools import Tool
from langchain.agents import initialize_agent, AgentType
import torch

# ========== CONFIGURATION ==========
LLAMA_MODEL_PATH = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DOC_PATH = "mlops_doc.txt"
DB_PATH = "mlops.db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ========== CUSTOM LLM ==========

class MyDirectLLM(LLM):
    def __init__(self, model, tokenizer, max_new_tokens=256, temperature=0.2):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

    @property
    def _llm_type(self):
        return "direct_llama"

    @property
    def _identifying_params(self):
        # For LangChain's internal validation and serialization
        return {
            "model": str(self.model.__class__),
            "tokenizer": str(self.tokenizer.__class__),
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
        }

    def _call(self, prompt, stop=None, **kwargs):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        with torch.no_grad():
            output = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        decoded = self.tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True)
        return decoded.strip()

# ========== LOAD LOCAL LLM ==========
print("Loading Llama 3.1 8B LLM...")
tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(LLAMA_MODEL_PATH).to("cpu")
llm = MyDirectLLM(model, tokenizer)

# ========== BUILD FAISS VECTOR STORE ==========
print("Building FAISS vector store from documentation...")
with open(DOC_PATH, "r", encoding="utf-8") as f:
    doc_text = f.read()
doc_chunks = [chunk.strip() for chunk in doc_text.split("\n\n") if chunk.strip()]

embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectorstore = FAISS.from_texts(doc_chunks, embedding_model)

# ========== SQL TOOL ==========
import re
def nl_to_sql(question):
    prompt = (
        "You are an expert SQL assistant. Given the schema: models(model_name, current_stage, version), "
        "write an SQLite SQL query for the question. "
        "Here is an example:\n"
        "Question: Which models are in production?\n"
        "SQL: SELECT * FROM models WHERE current_stage = 'production';\n"
        f"Question: {question}\n"
        "SQL:"
    )
    output = llm(prompt).strip()
    # Try to extract SQL using regex
    match = re.search(r"(SELECT .*?;)", output, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1)
    # Fallback: scan lines
    for line in output.split('\n'):
        if line.strip().lower().startswith("select"):
            return line.strip()
    return output

def execute_sql(sql):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    try:
        cursor.execute(sql)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        conn.close()
        if columns:
            return "\n".join([", ".join(columns)] + [", ".join(map(str, row)) for row in rows])
        else:
            return "\n".join(map(str, rows))
    except Exception as e:
        conn.close()
        return f"SQL Error: {e}"

def sql_tool_func(question: str) -> str:
    sql = nl_to_sql(question)
    print(f"[Generated SQL]: {sql}")
    result = execute_sql(sql)
    return f"SQL Result:\n{result}"

sql_tool = Tool(
    name="Model SQL Tool",
    func=sql_tool_func,
    description=(
        "Use this tool to answer questions about models in production, their names, versions, or stages. "
        "Input should be a natural language question about the production models."
    )
)

# ========== RETRIEVAL TOOL ==========
def retrieval_tool_func(query: str) -> str:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.get_relevant_documents(query)
    context = "\n---\n".join([doc.page_content for doc in docs])
    prompt = (
        "You are an MLOps assistant. Use the following documentation to answer the question.\n"
        f"Documentation:\n{context}\n"
        f"Question: {query}\n"
        "Answer:"
    )
    answer = llm(prompt)
    return answer.strip()

retrieval_tool = Tool(
    name="MLOps Documentation Tool",
    func=retrieval_tool_func,
    description=(
        "Use this tool to answer general MLOps questions based on documentation. "
        "Input should be a natural language question about MLOps practices, procedures, or concepts."
    )
)

# ========== AGENT SETUP ==========
tools = [sql_tool, retrieval_tool]
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# ========== MAIN LOOP ==========
if __name__ == "__main__":
    print("MLOps Assistant (LangChain, Direct LLM) is ready. Type your question (Ctrl+C to exit):")
    while True:
        try:
            user_query = input("\n> ")
            response = agent.run(user_query)
            print(f"\n{response}\n")
        except KeyboardInterrupt:
            print("\nExiting.")
            break
