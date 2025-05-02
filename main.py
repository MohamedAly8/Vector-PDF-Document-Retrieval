from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

load_dotenv()
import os

# --- Load Environment Variables ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY not found in environment variables.")

# --- Define Paths ---
PDF_FOLDER_PATH = "pdf_documents"
VECTORSTORE_PATH = "my_faiss_index"

# --- Indexing Phase ---
def create_index():
    print(f"Loading documents from: {PDF_FOLDER_PATH}")
    loader = DirectoryLoader(PDF_FOLDER_PATH, glob="**/*.pdf")
    documents = loader.load()
    print(f"Loaded {len(documents)} documents.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} chunks.")

    print("Creating embedding model...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)

    print("Creating and saving vector store...")
    # Ensure the directory exists before saving
    os.makedirs(VECTORSTORE_PATH, exist_ok=True)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTORSTORE_PATH)
    print(f"Vector store saved to: {VECTORSTORE_PATH}")

# --- Query Phase ---
def find_relevant_pdfs_with_score_and_chunks(query):
    if not os.path.exists(VECTORSTORE_PATH):
        print(f"Vector store not found at {VECTORSTORE_PATH}. Please run the indexing first.")
        return {}

    print("Loading embedding model...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=OPENAI_API_KEY)

    print("Loading vector store...")
    vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)

    print(f"Searching for documents relevant to: '{query}'")
    results_with_scores = vectorstore.similarity_search_with_score(query, k=5) # Adjust k as needed

    relevant_pdfs_with_chunks = {}
    if results_with_scores:
        print("\n--- Most Relevant Chunks Found with Scores ---")
        for i, (doc, score) in enumerate(results_with_scores):
            source = doc.metadata.get('source', 'Unknown Source')
            pdf_filename = os.path.basename(source)
            chunk_content = doc.page_content[:150] + "..." # Display a snippet
            page_number = doc.metadata.get('page', 'N/A')

            if pdf_filename not in relevant_pdfs_with_chunks:
                relevant_pdfs_with_chunks[pdf_filename] = {
                    "best_score": score,
                    "chunks": []
                }
            # Update best score if a more relevant chunk is found
            relevant_pdfs_with_chunks[pdf_filename]["best_score"] = min(relevant_pdfs_with_chunks[pdf_filename]["best_score"], score)
            relevant_pdfs_with_chunks[pdf_filename]["chunks"].append({
                "content": chunk_content,
                "score": f"{score:.4f}",
                "page": page_number
            })
            print(f"Chunk {i+1}: Source: {source}, Page: {page_number}, Similarity Score: {score:.4f}")
            print(f"   Content: {chunk_content}")
        print("--- End Chunks ---")

    # Sort PDFs by their best score
    sorted_pdfs_with_chunks = sorted(relevant_pdfs_with_chunks.items(), key=lambda item: item[1]["best_score"])
    return sorted_pdfs_with_chunks