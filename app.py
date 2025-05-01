from flask import Flask, request, render_template
import os
from main import create_index, find_relevant_pdfs_with_score_and_chunks
import threading

app = Flask(__name__)

isIndexing = False
indexing_thread = None

PDF_FOLDER_PATH = "pdf_documents"
if not os.path.exists(PDF_FOLDER_PATH):
    os.makedirs(PDF_FOLDER_PATH)

VECTORSTORE_PATH = "my_faiss_index"

def start_indexing():
    global isIndexing
    global indexing_thread
    if not os.path.exists(VECTORSTORE_PATH):
        print("Vector store not found. Starting indexing in background...")
        isIndexing = True
        indexing_thread = threading.Thread(target=create_index)
        indexing_thread.start()
        print("Indexing started in background.")
    else:
        print("Vector store already exists.")

start_indexing() # Start indexing when the app starts

@app.route("/", methods=["GET", "POST"])
def index():
    global isIndexing
    global indexing_thread

    if isIndexing:
      if indexing_thread and indexing_thread.is_alive():
        return render_template("loading.html")
      else:
        isIndexing = False
        return render_template("index.html", results=None)

    results_with_chunks = []
    if request.method == "POST":
        query = request.form["query"]
        if query:
            relevant_pdfs_data = find_relevant_pdfs_with_score_and_chunks(query)
            for filename, data in relevant_pdfs_data:
                results_with_chunks.append({
                    "filename": filename,
                    "best_score": f"{data['best_score']:.4f}",
                    "chunks": data["chunks"]
                })
    return render_template("index.html", results=results_with_chunks)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=3333, debug=True)