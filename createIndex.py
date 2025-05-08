from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os
import logging

# --- Set up logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Load Environment Variables ---
def _load_api_key():
    """
    Loads the OpenAI API key from the environment variables.
    Prints a warning message if the key is not found.
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.warning("OPENAI_API_KEY not found in environment variables.")
    return api_key

# --- Define Paths ---
PDF_FOLDER_PATH = "pdf_documents"
VECTORSTORE_PATH = "my_faiss_index"

# --- Document Loading and Splitting ---
def _load_documents(folder_path, glob_pattern="**/*.pdf"):
    """
    Loads documents from a specified directory.

    Args:
        folder_path (str): The path to the directory containing the documents.
        glob_pattern (str, optional):  The glob pattern to match files. Defaults to "**\/*.pdf".

    Returns:
        list: A list of Langchain Document objects.
    """
    logging.info(f"Loading documents from: {folder_path}")
    loader = DirectoryLoader(folder_path, glob=glob_pattern)
    documents = loader.load()
    logging.info(f"Loaded {len(documents)} documents.")
    return documents

def _split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    Splits the loaded documents into smaller chunks for better processing.

    Args:
        documents (list): A list of Langchain Document objects to split.
        chunk_size (int, optional): The size of each text chunk. Defaults to 1000.
        chunk_overlap (int, optional): The overlap between adjacent chunks. Defaults to 200.

    Returns:
        list: A list of Langchain Document objects representing the text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)
    logging.info(f"Split documents into {len(chunks)} chunks.")
    return chunks

# --- Vector Store Creation ---
def _create_embeddings(api_key, model_name="text-embedding-3-large"):
    """
    Creates an OpenAI embeddings model.

    Args:
        api_key (str): The OpenAI API key.
        model_name (str, optional): The name of the embedding model to use.
            Defaults to "text-embedding-3-large".

    Returns:
        OpenAIEmbeddings: An instance of the OpenAIEmbeddings class.
    """
    logging.info("Creating embedding model...")
    embeddings = OpenAIEmbeddings(model=model_name, openai_api_key=api_key)
    return embeddings

def _create_vectorstore(chunks, embeddings, vectorstore_path):
    """
    Creates and saves a FAISS vector store from the document chunks and embeddings.

    Args:
        chunks (list): A list of Langchain Document objects representing the text chunks.
        embeddings (OpenAIEmbeddings): An instance of the OpenAIEmbeddings class.
        vectorstore_path (str): The path where the vector store should be saved.
    """
    logging.info("Creating and saving vector store...")
    os.makedirs(vectorstore_path, exist_ok=True)  # Ensure the directory exists
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(vectorstore_path)
    logging.info(f"Vector store saved to: {vectorstore_path}")
    return vectorstore

def create_index():
    """
    Creates the FAISS index from the PDF documents in the specified folder.
    This function orchestrates the document loading, splitting, embedding,
    and vector store creation processes.
    """
    api_key = _load_api_key()
    documents = _load_documents(PDF_FOLDER_PATH)
    chunks = _split_documents(documents)
    embeddings = _create_embeddings(api_key)
    _create_vectorstore(chunks, embeddings, VECTORSTORE_PATH)