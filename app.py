from flask import Flask, request, render_template
import os
from createIndex import create_index
from documentSearch import find_relevant_pdfs_with_score_and_chunks
import threading
import logging

app = Flask(__name__)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

isIndexing = False
indexing_thread = None

PDF_FOLDER_PATH = "/app/pdf_documents"
if not os.path.exists(PDF_FOLDER_PATH):
    os.makedirs(PDF_FOLDER_PATH)

VECTORSTORE_PATH = "/app/my_faiss_index"

def start_indexing():
    global isIndexing
    global indexing_thread
    # if the vector store path is empty 
    if not os.listdir(VECTORSTORE_PATH):
        print("Vector store not found. Starting indexing in background...")
        isIndexing = True
        indexing_thread = threading.Thread(target=create_index)
        indexing_thread.start()
        print("Indexing started in background.")
    else:
        print("Vector store already exists.")

def renderLoading():
    return render_template("loading.html")


start_indexing() 

@app.route("/", methods=["GET", "POST"])
def index():
    global isIndexing
    global indexing_thread

    if isIndexing:
      if indexing_thread and indexing_thread.is_alive():
        renderLoading()
      else:
        isIndexing = False
        return render_template("index.html", results=None)

    results_with_chunks = []

    query = ""
    
    if request.method == "POST":
        query = request.form["query"]

        app.logger.debug(f"Query received: {query}")

        if query:
            app.logger.debug("Finding relevant PDFs...")
            relevant_pdfs_data = find_relevant_pdfs_with_score_and_chunks(query)
            app.logger.debug(f"Relevant PDFs found: {relevant_pdfs_data}")
            for filename, data in relevant_pdfs_data:
                results_with_chunks.append({
                    "filename": filename,
                    "best_score": f"{data['best_score']:.4f}",
                    "chunks": data["chunks"]
                })
    return render_template("index.html", results=results_with_chunks, query=query)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3333, debug=True)