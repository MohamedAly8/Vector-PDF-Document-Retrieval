from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import os
import logging

# --- Set up logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Define Paths ---
VECTORSTORE_PATH = "my_faiss_index"

def _load_vectorstore(vectorstore_path, embeddings):
    """
    Loads the FAISS vector store from the specified path.

    Args:
        vectorstore_path (str): The path to the saved vector store.
        embeddings (OpenAIEmbeddings):  An instance of the OpenAIEmbeddings.

    Returns:
        FAISS: The loaded FAISS vector store, or None if the path doesn't exist.
    """
    if not os.path.exists(vectorstore_path):
        logging.error(f"Vector store not found at {vectorstore_path}. Please run the indexing first.")
        return None
    logging.info("Loading vector store...")
    vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
    return vectorstore

def _search_vectorstore(vectorstore, query, k=20):
    """
    Searches the vector store for documents relevant to the given query.

    Args:
        vectorstore (FAISS): The FAISS vector store to search.
        query (str): The query string.
        k (int, optional): The number of top results to return. Defaults to 5.

    Returns:
        list: A list of tuples, where each tuple contains a Langchain Document
              and its similarity score.  Returns an empty list if vectorstore is None.
    """
    if vectorstore is None:
        return []

    logging.info(f"Searching for documents relevant to: '{query}'")
    results_with_scores = vectorstore.similarity_search_with_score(query, k=k)
    return results_with_scores

def _process_search_results(results_with_scores):
    """
    Processes the search results, organizing them by PDF filename and extracting
    relevant information from each chunk.

    Args:
        results_with_scores (list): A list of tuples, where each tuple contains a
            Langchain Document and its similarity score.

    Returns:
        dict: A dictionary where keys are PDF filenames, and values are dictionaries
              containing the best score and a list of relevant chunks. Returns an empty dict if results is empty.
    """
    relevant_pdfs_with_chunks = {}
    if results_with_scores:
        logging.info("\n--- Most Relevant Chunks Found with Scores ---")
        for i, (doc, score) in enumerate(results_with_scores):
            source = doc.metadata.get('source', 'Unknown Source')
            pdf_filename = os.path.basename(source)
            chunk_content = doc.page_content[:150] + "..."  # Display a snippet
            page_number = doc.metadata.get('page', 'N/A')

            if pdf_filename not in relevant_pdfs_with_chunks:
                relevant_pdfs_with_chunks[pdf_filename] = {
                    "best_score": score,
                    "chunks": []
                }
            # Update best score if a more relevant chunk is found
            relevant_pdfs_with_chunks[pdf_filename]["best_score"] = min(
                relevant_pdfs_with_chunks[pdf_filename]["best_score"], score
            )
            relevant_pdfs_with_chunks[pdf_filename]["chunks"].append({
                "content": chunk_content,
                "score": f"{score:.4f}",
                "page": page_number
            })
            logging.info(f"Chunk {i + 1}: Source: {source}, Page: {page_number}, Similarity Score: {score:.4f}")
            logging.info(f"   Content: {chunk_content}")
        logging.info("--- End Chunks ---")
    return relevant_pdfs_with_chunks

def find_relevant_pdfs_with_score_and_chunks(query):
    """
    Finds PDFs relevant to the given query, along with their similarity scores and relevant chunks.

    Args:
        query (str): The query string.

    Returns:
         dict: A dictionary where keys are PDF filenames, and values are dictionaries
              containing the best score and a list of relevant chunks, sorted by best score.
              Returns an empty dict if no relevant PDFs are found or the vectorstore doesn't exist.
    """
    from dotenv import load_dotenv
    import os
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", openai_api_key=api_key)
    vectorstore = _load_vectorstore(VECTORSTORE_PATH, embeddings)
    results_with_scores = _search_vectorstore(vectorstore, query)
    relevant_pdfs_with_chunks = _process_search_results(results_with_scores)

    # Sort PDFs by their best score
    sorted_pdfs_with_chunks = sorted(
        relevant_pdfs_with_chunks.items(), key=lambda item: item[1]["best_score"]
    )
    return sorted_pdfs_with_chunks