#!/bin/bash

# Prompt the user for their OpenAI API key
read -p "Enter your OpenAI API key: " OPENAI_API_KEY

# Determine the absolute path to the directory containing PDF documents
PDF_DIR="$PWD/pdf_documents"

# Determine the absolute path for the vector index (optional, for persistence)
INDEX_DIR="$PWD/my_faiss_index"

# Check if the PDF directory exists
if [ ! -d "$PDF_DIR" ]; then
  mkdir -p "$PDF_DIR"
  echo "Created directory: $PDF_DIR. Please place your PDF files here."
  read -p "Press Enter once you've placed your PDFs in '$PDF_DIR'..."
fi

# Run the Docker container with volume mounts
docker run -p 3333:3333 \
  -e OPEN_AI_KEY="$OPENAI_API_KEY" \
  -v "$PDF_DIR:/app/pdf_documents" \
  -v "$INDEX_DIR:/app/my_faiss_index" \
  vectors