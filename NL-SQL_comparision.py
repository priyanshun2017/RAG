import os
import sqlite_utils
import pandas as pd
from sqlalchemy import create_engine
from typing import List, Dict, Any

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# LangChain NL-to-SQL Agent Imports
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents import AgentExecutor
from langchain_community.llms import Ollama
from langchain.agents.agent_types import AgentType


# --- Configuration ---
PDF_PATH = "nvme_spec.pdf"
CHROMA_DB_PATH = "./nvme_chroma_db"
SQL_DB_PATH = "nvme_structured.db"

# Ollama Models
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
OLLAMA_RAG_MODEL = "llama2"     # General purpose LLM for RAG answers
OLLAMA_SQL_MODEL = "sqlcoder"   # Specialized LLM for NL-to-SQL generation

# ==============================================================================
# 1. PATH 1: VECTOR DATABASE (RAG) FUNCTIONS
# ==============================================================================

def ingest_data_to_vector_db():
    """Loads the PDF, chunks it, embeds it, and saves it to Chroma DB."""
    if not os.path.exists(PDF_PATH):
        print(f"🚨 Error: PDF file not found at {PDF_PATH}. Skipping Vector DB Ingestion.")
        return

    print("--- 🛠️ Starting Vector DB Ingestion ---")
    try:
        # 1. Load the document
        loader = PyPDFLoader(PDF_PATH)
        data = loader.load()

        # 2. Split the document
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,  # Increased chunk size for complex specs
            chunk_overlap=200
        )
        docs = text_splitter.split_documents(data)
        print(f"Loaded {len(data)} pages. Split into {len(docs)} chunks.")

        # 3. Initialize Ollama Embeddings
        embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)

        # 4. Create the Vector Store (Chroma)
        Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=CHROMA_DB_PATH
        )
        print(f"Vector database created and persisted at {CHROMA_DB_PATH}")

    except Exception as e:
        print(f"An error occurred during Vector DB ingestion: {e}")

def query_vector_db(question: str):
    """Performs RAG query against the Vector Database."""
    print(f"\n--- 🔎 Querying Vector DB with RAG (Model: {OLLAMA_RAG_MODEL}) ---")
    
    if not os.path.exists(CHROMA_DB_PATH):
        print("Vector DB not found. Please run ingestion first.")
        return

    try:
        # Re-initialize embeddings and vector store
        embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)
        vectorstore = Chroma(persist_directory=CHROMA_DB_PATH, embedding_function=embeddings)

        # Initialize the LLM
        llm = ChatOllama(model=OLLAMA_RAG_MODEL, temperature=0.1)

        # Define the RAG prompt template
        template = """
        You are an expert on the NVMe specification. Use the following context 
        to accurately and concisely answer the user's question. 
        If you cannot find the answer in the context, state clearly that the information is not available in the specification.

        Context: {context}
        Question: {question}

        Answer:
        """
        QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

        # Create the RAG chain
        qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
            return_source_documents=True
        )

        result = qa_chain.invoke({"query": question})
        print("\n[RAG Answer]")
        print(result["result"])
        
        print("\n[Top Source Documents Used]")
        for doc in result["source_documents"][:2]: 
            # Note: page metadata is 0-indexed, so add 1
            page_num = doc.metadata.get('page', 'N/A')
            print(f"- Page {page_num + 1 if isinstance(page_num, int) else page_num}: {doc.page_content[:100]}...")
        print("--------------------------------------------------")

    except Exception as e:
        print(f"An error occurred during Vector DB query: {e}")


# ==============================================================================
# 2. PATH 2: SQL DATABASE (NL-TO-SQL) FUNCTIONS
# ==============================================================================

def create_and_populate_sql_db():
    """Creates a conceptual SQLite database for structured NVMe data."""
    print("--- Starting SQL DB Structuring and Population ---")
    try:
        db = sqlite_utils.Database(SQL_DB_PATH)

        # --- Table 1: NVMe OpCodes (Conceptual Example) ---
        # NOTE: In a real-world app, this data would be scraped from the spec
        opcodes_data = [
            {"opcode_id": 0x00, "name": "Flush", "admin_command": 1, "description": "Ensures all data is written to NVM."},
            {"opcode_id": 0x01, "name": "Identify", "admin_command": 1, "description": "Returns controller or namespace information."},
            {"opcode_id": 0x02, "name": "AER", "admin_command": 1, "description": "Admin Event Request."},
            {"opcode_id": 0x05, "name": "Write", "admin_command": 0, "description": "Writes data to a namespace."},
            {"opcode_id": 0x09, "name": "Dataset Management", "admin_command": 0, "description": "Indicates to the controller sets of logical blocks that are no longer in use."},
        ]
        db["opcodes"].insert_all(opcodes_data, pk="opcode_id", replace=True)

        # --- Table 2: NVMe Registers (Conceptual Example) ---
        registers_data = [
            {"register_name": "CAP", "offset": "0000h", "size_bytes": 8, "type": "Controller Capability"},
            {"register_name": "VS", "offset": "0008h", "size_bytes": 4, "type": "Version"},
            {"register_name": "CC", "offset": "0014h", "size_bytes": 4, "type": "Controller Configuration"},
            {"register_name": "AQA", "offset": "0024h", "size_bytes": 4, "type": "Admin Queue Attributes"},
        ]
        db["registers"].insert_all(registers_data, pk="register_name", replace=True)

        print(f"SQL database created and populated at {SQL_DB_PATH}.")
        print(f"Available tables: {db.table_names()}")
    
    except Exception as e:
        print(f"An error occurred during SQL DB population: {e}")


def query_sql_db(question: str):
    """Uses a specialized LLM to convert NL to SQL, execute it, and summarize the result."""
    print(f"\n--- Querying SQL DB with NL-to-SQL (Model: {OLLAMA_SQL_MODEL}) ---")
    
    if not os.path.exists(SQL_DB_PATH):
        print("SQL DB not found. Please run structuring and population first.")
        return

    try:
        # 1. Initialize SQL Database connection
        db = SQLDatabase.from_uri(f"sqlite:///{SQL_DB_PATH}")

        # 2. Initialize the LLM (sqlcoder is typically best for generating the SQL)
        llm = Ollama(model=OLLAMA_SQL_MODEL, temperature=0)

        # 3. Create the SQL Agent Toolkit
        # The toolkit provides the LLM with the ability to:
        # - List tables (tool: list_tables)
        # - Get table schema (tool: schema_sql)
        # - Execute SQL (tool: query_sql)
        toolkit = SQLDatabaseToolkit(db=db, llm=llm)

        # 4. Create the SQL Agent
        agent_executor = AgentExecutor.from_agent_and_toolkit(
            # Using ZERO_SHOT_REACT_DESCRIPTION agent type for a step-by-step thinking process
            agent=toolkit.create_agent(AgentType.ZERO_SHOT_REACT_DESCRIPTION, llm=llm),
            toolkit=toolkit,
            verbose=False # Set to True to see the LLM's thought process/steps
        )

        response = agent_executor.invoke({"input": question})
        print("\n[NL-to-SQL Answer]")
        print(response["output"])
        print("--------------------------------------------------")
        
    except Exception as e:
        print(f"An error occurred during NL-to-SQL execution. See verbose output above for steps. Error: {e}")


# ==============================================================================
# 3. MAIN EXECUTION AND COMPARISON
# ==============================================================================

def main():
    """Main function to setup databases and run comparative queries."""

    print("--- NVMe NL-to-Database Comparison App 🚀 ---")
    
    # --- Data Setup Step (Run these first) ---
    # 1. Setup Vector DB
    ingest_data_to_vector_db() 

    # 2. Setup SQL DB
    create_and_populate_sql_db()
    
    # UNCOMMENT THE TWO LINES ABOVE AFTER YOU HAVE THE nvme_spec.pdf 
    # AND ARE READY TO INGEST/STRUCTURE DATA. 
    # COMMENT THEM OUT AFTER THE FIRST SUCCESSFUL RUN.
    
    if not os.path.exists(CHROMA_DB_PATH) or not os.path.exists(SQL_DB_PATH):
         print("\nPlease UNCOMMENT the 'ingest_data_to_vector_db()' and 'create_and_populate_sql_db()' lines in main() to set up the databases first. Then comment them out and re-run.")
         return

    print("\n--- Databases are ready. Running comparative queries... ---")

    # --- Comparative Test Queries ---

    # Query 1: Unstructured/RAG-suited question (Requires context from the body of the 700-page spec)
    rag_question = "What are the power management state transition rules for a non-operational controller, and which section describes them?"
    
    # Query 2: Structured/SQL-suited question (Requires querying structured tables)
    sql_question = "Which NVMe OpCodes are Admin Commands, and list their OpCode ID and Name."

    # Query 3: Hybrid question (Requires combining both structured and unstructured knowledge)
    hybrid_question = "What is the offset of the Controller Configuration (CC) register, and what are the fields within it that define the Arbitration Mechanism?"

    
    # --------------------------------------------------------------------------
    # TEST 1: RAG's Strength (Conceptual/Unstructured Query)
    # --------------------------------------------------------------------------
    print("\n\n==================================================")
    print("TEST 1: CONCEPTUAL/UNSTRUCTURED QUERY")
    print(f"Q: {rag_question}")
    query_vector_db(rag_question)
    query_sql_db(rag_question) # SQL path expected to fail or return an irrelevant schema-based answer


    # --------------------------------------------------------------------------
    # TEST 2: NL-to-SQL's Strength (Fact-based/Structured Query)
    # --------------------------------------------------------------------------
    print("\n\n==================================================")
    print("TEST 2: FACT-BASED/STRUCTURED QUERY")
    print(f"Q: {sql_question}")
    query_sql_db(sql_question)
    query_vector_db(sql_question) # RAG path expected to be slow or less precise/structured


    # --------------------------------------------------------------------------
    # TEST 3: Hybrid Query (Tests both for their ability to retrieve data)
    # --------------------------------------------------------------------------
    print("\n\n==================================================")
    print("TEST 3: HYBRID QUERY")
    print(f"Q: {hybrid_question}")
    query_sql_db(hybrid_question) # Will only answer the register offset part
    query_vector_db(hybrid_question) # Will likely be better at answering the 'fields' part

# Execute the main function
if __name__ == "__main__":
    main()
