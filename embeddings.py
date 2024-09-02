import os
import sys
import logging
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from PyPDF2 import PdfReader
from dotenv import load_dotenv

# Load environment variables from .env file if needed
load_dotenv()

# Configure logging for better error handling
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Define the embedding model
embedding_model_id = "BAAI/bge-small-en-v1.5"
embeddings = HuggingFaceEmbeddings(model_name=embedding_model_id)

def get_pdf_text(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    try:
        pdf_reader = PdfReader(pdf_path)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
            else:
                logging.warning(f"No text extracted from page {page}")
    except Exception as e:
        logging.error(f"Error reading PDF file {pdf_path}: {e}")
    return text

def get_text_chunks(text_data):
    """Split text into manageable chunks."""
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = []
    try:
        chunks = text_splitter.split_text(text_data)
    except Exception as e:
        logging.error(f"Error splitting text into chunks: {e}")
    return chunks

def save_embeddings_to_faiss(documents, embeddings, index_path="./doc_db/faiss_index"):
    """Save embeddings to a FAISS vector store."""
    try:
        # Create FAISS index from documents
        embeddings_db = FAISS.from_documents(documents, embeddings)
        # Ensure the directory exists
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        # Save the FAISS index locally
        embeddings_db.save_local(index_path)
        logging.info("FAISS index saved successfully at %s", index_path)
    except Exception as e:
        logging.error(f"Error creating or saving FAISS index: {e}")

def main(pdf_path):
    # Load data from the PDF
    text_data = get_pdf_text(pdf_path)

    if not text_data:
        logging.error("No text extracted from the PDF. Please check the file path or content.")
        sys.exit(1)

    # Split the data into chunks
    all_chunks = get_text_chunks(text_data)

    # Wrap chunks in Document objects
    documents = [Document(page_content=chunk) for chunk in all_chunks]

    # Save embeddings to FAISS vector store
    save_embeddings_to_faiss(documents, embeddings)

if __name__ == "__main__":
    # Correct path to your PDF file
    pdf_path = "./docs/kenyan_constitution.pdf"
    main(pdf_path)
