import sqlite3
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.tools import Tool
from langchain.agents import initialize_agent, AgentType
import re

# ========== CONFIGURATION ==========
LLAMA_MODEL_PATH = "meta-llama/Meta-Llama-3.1-8B-Instruct"  # Update to your local path if needed
DOC_PATH = "mlops_doc.txt"
DB_PATH = "mlops.db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# ========== LOAD LOCAL LLM WITH PIPELINE ==========
print("Loading Llama 3.1 8B LLM...")
tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(LLAMA_MODEL_PATH)
llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=256,
    temperature=0.2,
    do_sample=True,
    device=-1  # CPU
)
llm = HuggingFacePipeline(pipeline=llm_pipeline)

# ========== BUILD FAISS VECTOR STORE ==========
print("Building FAISS vector store from documentation...")
with open(DOC_PATH, "r", encoding="utf-8") as f:
    doc_text = f.read()
doc_chunks = [chunk.strip() for chunk in doc_text.split("\n\n") if chunk.strip()]
embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectorstore = FAISS.from_texts(doc_chunks, embedding_model)

# ========== SQL TOOL ==========

def nl_to_sql(question):
    prompt = (
        "Given the following SQLite table schema:\n"
        "models(model_name, current_stage, version)\n"
        "Convert the following natural language question to a syntactically correct SQL query. "
        "Return ONLY the SQL query, nothing else.\n"
        "Example:\n"
        "Question: Which models are in production?\n"
        "SELECT * FROM models WHERE current_stage = 'production';\n"
        f"Question: {question}\n"
    )
    output = llm(prompt).strip()
    # Remove any line that exactly matches the question
    output_lines = [line.strip() for line in output.split('\n') if line.strip() and line.strip() != f"Question: {question}"]
    # Look for a line starting with SQL keywords and ending with a semicolon
    for line in output_lines:
        if re.match(r"^(SELECT|INSERT|UPDATE|DELETE|WITH)\b.*;", line, re.IGNORECASE):
            return line
    # Try to find a SQL statement anywhere in the output
    match = re.search(r"(SELECT|INSERT|UPDATE|DELETE|WITH)[\s\S]+?;", output, re.IGNORECASE)
    if match:
        return match.group(0).strip()
    # If nothing found, return empty string or error
    return ""


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
    print("MLOps Assistant (LangChain, HuggingFacePipeline) is ready. Type your question (Ctrl+C to exit):")
    while True:
        try:
            user_query = input("\n> ")
            response = agent.run(user_query)
            print(f"\n{response}\n")
        except KeyboardInterrupt:
            print("\nExiting.")
            break
