import os
import glob
import uuid
from typing import List
from PIL import Image

# LangChain Imports
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage
from langchain.storage import LocalFileStore
from langchain.retrievers.multi_vector import MultiVectorRetriever

# --- CONFIGURATION ---
OLLAMA_EMBEDDING_MODEL = "gemma3:12b"
OLLAMA_CAPTION_MODEL = "gemma3:12b"
OLLAMA_GENERATION_MODEL = "gpt-oss:20b"
VECTOR_DB_PATH = "./chroma_db"
DOCSTORE_PATH = "./docstore"

TRANSCRIPT_PATH = "transcript.txt"
KB_TEXT_DIR = "knowledge_base/text/"
KB_IMAGES_DIR = "knowledge_base/images/"

# Initialize the Document Store for raw image paths/data
# This stores the original image path/file, indexed by a unique ID (doc_id)
docstore = LocalFileStore(DOCSTORE_PATH)


# ====================================================
# PHASE 1: Multimodal Indexing and Document Preparation
# ====================================================

def generate_image_documents(image_dir: str) -> List[Document]:
    """Uses the multimodal LLM to describe images and prepares documents for indexing."""
    print("Starting image captioning with:", OLLAMA_CAPTION_MODEL)
    
    # Initialize the multimodal LLM
    multimodal_llm = ChatOllama(model=OLLAMA_CAPTION_MODEL)
    
    image_paths = glob.glob(os.path.join(image_dir, '*'))
    image_docs = []
    
    prompt = "Describe this image in detail, noting any text, charts, or key concepts. Provide a concise summary optimized for semantic retrieval, and ensure you mention the key topic discussed."
    
    # List to store (doc_id, image_path) tuples for the docstore
    docstore_entries = []

    for path in image_paths:
        if os.path.isfile(path) and path.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"  - Processing {os.path.basename(path)}...")
            doc_id = str(uuid.uuid4())
            
            try:
                # 1. Generate the text caption/summary using the multimodal model
                response = multimodal_llm.invoke(
                    [HumanMessage(content=prompt)],
                    images=[path]
                )
                
                # 2. Create the caption document for the vector store
                caption = f"IMAGE CAPTION for {os.path.basename(path)}:\n" + response.content
                image_docs.append(
                    Document(
                        page_content=caption,
                        metadata={
                            "doc_id": doc_id,
                            "source": path,
                            "type": "image_caption",
                            "filename": os.path.basename(path)
                        }
                    )
                )
                
                # 3. Prepare the entry for the LocalFileStore (key=doc_id, value=image_path)
                docstore_entries.append((doc_id, path))

            except Exception as e:
                print(f"    [ERROR] Could not process {path}: {e}")
                
    # Batch storage of the raw image paths in the docstore
    if docstore_entries:
        docstore.mset(docstore_entries)
    
    print(f"Finished. Generated {len(image_docs)} documents from images and stored paths in docstore.")
    return image_docs

def load_and_chunk_documents():
    """Loads all text and caption documents and splits them."""
    # Load Transcript
    transcript_loader = TextLoader(TRANSCRIPT_PATH)
    transcript_docs = transcript_loader.load()
    for doc in transcript_docs:
        doc.metadata.update({"doc_id": str(uuid.uuid4()), "source": TRANSCRIPT_PATH, "type": "transcript"})

    # Load KB Text
    kb_loader = DirectoryLoader(KB_TEXT_DIR, glob="**/*.txt", loader_cls=TextLoader)
    kb_text_docs = kb_loader.load()
    for doc in kb_text_docs:
        doc.metadata.update({"doc_id": str(uuid.uuid4()), "type": "kb_text"})

    # Load KB Image Descriptions (which also populate the docstore)
    kb_image_docs = generate_image_documents(KB_IMAGES_DIR)

    # Combine all documents (captions, transcript, text KB)
    all_documents = transcript_docs + kb_text_docs + kb_image_docs

    # Text Splitting
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunked_documents = text_splitter.split_documents(all_documents)
    print(f"Split {len(all_documents)} documents into {len(chunked_documents)} chunks for embedding.")
    
    return chunked_documents


# ====================================================
# PHASE 2: Retrieval Setup (Multi-Vector Retriever)
# ====================================================

def setup_retriever(chunked_documents: List[Document]) -> MultiVectorRetriever:
    """Sets up the Ollama embeddings and the MultiVectorRetriever."""
    print("\nStarting indexing and vector store creation...")
    
    # 1. Define Embeddings
    embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL)

    # 2. Create Vector Store (Chroma)
    vectorstore = Chroma.from_documents(
        documents=chunked_documents,
        embedding=embeddings,
        persist_directory=VECTOR_DB_PATH
    )
    print(f"Successfully created Vector Store at {VECTOR_DB_PATH}")

    # 3. Define the MultiVector Retriever
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=docstore,
        id_key="doc_id"  # Key used to lookup the raw document (image path)
    )
    return retriever


# ====================================================
# PHASE 3: Report Generation Chain
# ====================================================

def format_multimodal_context(retrieved_docs: List[Document]) -> str:
    """Formats the retrieved text and image paths into a single context string."""
    text_content = []
    image_references = []
    
    for doc in retrieved_docs:
        # Check if the document in the retrieved context came from an image caption
        if doc.metadata.get('type') == 'image_caption':
            # The docstore retrieval step returns the original document (the image path string in this case)
            image_path = docstore.mget([doc.metadata['doc_id']])[0] 
            
            # Use the caption (doc.page_content) to inform the LLM, but provide the path too
            image_references.append(
                f"Image Reference: {doc.metadata.get('filename', 'N/A')}\n"
                f"  - **Path:** {image_path}\n"
                f"  - **Caption/Relevance:** {doc.page_content}"
            )
        else:
            # All other text (transcript, KB text)
            text_content.append(doc.page_content)

    final_context = (
        "--- TRANSCRIPT & KNOWLEDGE BASE TEXTUAL CONTEXT ---\n" + "\n\n".join(text_content) +
        "\n\n--- IMAGE DATA RETRIEVED ---\n" + "\n".join(image_references)
    )
    return final_context


def generate_report(retriever: MultiVectorRetriever):
    """Defines and runs the final RAG chain."""
    
    # 1. Define LLM for Generation
    llm = ChatOllama(model=OLLAMA_GENERATION_MODEL, temperature=0.2)

    # 2. Define the Report Prompt
    template = """
    You are an expert analyst and report generator. Your task is to generate a comprehensive, structured, and insightful report.

    The report must be based on the user's main request and the context provided below. The context is comprised of text segments and descriptive image captions/paths.

    Your report should:
    1.  **Synthesize** all information, prioritizing the 'transcript' and supporting it with the 'knowledge base' and image data.
    2.  **Integrate** findings from image captions, and when citing visual evidence, mention the **Image Filename/Path** from the "IMAGE DATA RETRIEVED" section to ground your findings.
    3.  **Use Markdown** for clear formatting (headings, bullet points, etc.).
    4.  Maintain a professional and objective tone.

    CONTEXT:
    {context}

    USER REQUEST:
    {question}

    COMPREHENSIVE REPORT:
    """
    report_prompt = ChatPromptTemplate.from_template(template)

    # 3. Define the RAG Chain (using LCEL)
    report_query = "Generate a full-scale analytical report on the main topics, conclusions, and supporting evidence found in the transcript and knowledge base. Focus on synthesizing key findings and identifying any potential action items or next steps. Ensure image sources are cited where appropriate."

    rag_chain = (
        # Retrieval returns the caption/summary for embedding matching
        {"context": retriever, "question": RunnablePassthrough()}
        # Custom formatting step retrieves the original image path using the doc_id
        | RunnablePassthrough.assign(context=format_multimodal_context)
        | report_prompt
        | llm
        | StrOutputParser()
    )

    print("\n\n" + "="*70)
    print(f"Generating Report using {OLLAMA_GENERATION_MODEL} with Multimodal Context...")
    print("="*70)

    # Run the RAG Chain
    final_report = rag_chain.invoke(report_query)

    print(final_report)
    print("="*70)


if __name__ == "__main__":
    # Create necessary directories if they don't exist
    os.makedirs(KB_TEXT_DIR, exist_ok=True)
    os.makedirs(KB_IMAGES_DIR, exist_ok=True)
    
    # --- DUMMY FILE CREATION (for testing) ---
    # Create dummy files if they don't exist, so the script can run
    if not os.path.exists(TRANSCRIPT_PATH):
        with open(TRANSCRIPT_PATH, "w") as f:
            f.write("Transcript: The primary focus of the Q4 strategy meeting was cost reduction in the supply chain, specifically a 15% target. There was significant discussion regarding the operational risk shown in the chart.")
        print(f"Created dummy file: {TRANSCRIPT_PATH}")
    if not os.path.exists(os.path.join(KB_TEXT_DIR, "doc1.txt")):
        with open(os.path.join(KB_TEXT_DIR, "doc1.txt"), "w") as f:
            f.write("Knowledge Base: The new vendor contracts effective Q1 introduce a 12% price hike for raw materials, directly challenging the Q4 cost reduction targets.")
        print(f"Created dummy file: {KB_TEXT_DIR}/doc1.txt")
    if not os.path.exists(os.path.join(KB_IMAGES_DIR, "chart_data.png")):
        # You'll need a real image file here for gemma3:12b to process. 
        # Create an empty file as a placeholder if you don't have one.
        # It's better to put a small PNG/JPG file in this location.
        try:
            Image.new('RGB', (100, 100), color = 'red').save(os.path.join(KB_IMAGES_DIR, "chart_data.png"))
            print(f"Created dummy image placeholder: {KB_IMAGES_DIR}/chart_data.png (Replace with actual image for best results!)")
        except Exception:
             print(f"Please manually place an image named 'chart_data.png' in {KB_IMAGES_DIR}")

    # --- MAIN EXECUTION ---
    try:
        chunked_documents = load_and_chunk_documents()
        retriever = setup_retriever(chunked_documents)
        generate_report(retriever)
    except Exception as e:
        print("\n[CRITICAL ERROR] The RAG chain failed to run. Check your Ollama server and model names.")
        print(f"Error details: {e}")
