import os
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

PDF_FOLDER_PATH = "pdf_documents"
VECTORSTORE_PATH = "my_faiss_index"
MODEL_NAME = "all-MiniLM-L6-v2"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 150

# --- Indexing
def create_index():
    print("Loading documents...")
    loader = DirectoryLoader(PDF_FOLDER_PATH, glob="**/*.pdf", loader_cls=PyMuPDFLoader, recursive=True, show_progress=True)
    documents = loader.load()
    if not documents:
        print(f"No PDF documents found in {PDF_FOLDER_PATH}")
        return

    print(f"Loaded {len(documents)} document sections.")

    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    docs_chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(docs_chunks)} chunks.")

    print("Initializing embedding model...")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Warning: OPENAI_API_KEY not found in environment variables.")
        return

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)

    print("Creating and saving vector store...")
    if not docs_chunks:
        print("No text chunks to index.")
        return

    vectorstore = FAISS.from_documents(docs_chunks, embeddings)
    vectorstore.save_local(VECTORSTORE_PATH)
    print(f"Vector store saved to {VECTORSTORE_PATH}")

# --- Query
def find_relevant_pdfs_with_score_and_chunks(query):
    if not os.path.exists(VECTORSTORE_PATH):
        print(f"Vector store not found at {VECTORSTORE_PATH}. Please run the indexing first.")
        return {}

    print("Loading embedding model...")
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        print("Warning: OPENAI_API_KEY not found in environment variables.")
        return {}

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", openai_api_key=openai_api_key)

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

if __name__ == "__main__":

    if os.path.exists(VECTORSTORE_PATH):
        user_prompt = input("Enter your query: ")
        if user_prompt:
            pdf_scores = find_relevant_pdfs_with_score(user_prompt)
            if pdf_scores:
                print("\n--- Most Relevant PDF Documents (with Similarity Scores) ---")
                for pdf, score in pdf_scores:
                    print(f"- {pdf}: Similarity Score = {score:.4f}")
            else:
                print("\nNo relevant PDF documents found for your query.")
    else:
         print(f"\nPlease create the index first by running the script (or the web app).")
         print(f"Ensure you have PDFs in the '{PDF_FOLDER_PATH}' directory.")