#!/bin/bash


if ! command -v docker &> /dev/null; then
  echo "Error: Docker is not installed or not in your PATH."
  echo "Please install Docker to run this application."
  exit 1
fi

PDF_DIR="$PWD/pdf_documents"
INDEX_DIR="$PWD/my_faiss_index"

# Check if the PDF directory exists
if [ ! -d "$PDF_DIR" ]; then
  mkdir -p "$PDF_DIR"
  echo "Created directory: $PDF_DIR. Please place your PDF files here."
  read -p "Press Enter once you've placed your PDFs in '$PDF_DIR'..."
fi

# Run container
docker run -p 3333:3333 \
  -e OPEN_AI_KEY="$OPENAI_API_KEY" \
  -v "$PDF_DIR:/app/pdf_documents" \
  -v "$INDEX_DIR:/app/my_faiss_index" \
  vectors


# Stop Container
CONTAINER_ID=$(docker ps -q --filter ancestor=vectors)

if [ -n "$CONTAINER_ID" ]; then
  echo "Stopping Docker container: $CONTAINER_ID"
  docker stop "$CONTAINER_ID"
  docker rm "$CONTAINER_ID"
else
  echo "No running Docker container found for the 'vectors' image."
fi