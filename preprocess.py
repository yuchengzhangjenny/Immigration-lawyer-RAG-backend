"""
Script to preprocess legal documents for the RAG system.
This script handles text cleaning, chunking, and embedding generation
for the downloaded legal documents.
"""

import os
import json
import re
import glob
import numpy as np
from tqdm import tqdm
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import pickle
import PyPDF2
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("preprocessor")

# Directories - using relative paths
DATA_DIR = "data"
CFR_DIR = os.path.join(DATA_DIR, "cfr_title8")
INA_DIR = os.path.join(DATA_DIR, "ina")
POLICY_DIR = os.path.join(DATA_DIR, "uscis_policy")
OUTPUT_DIR = os.path.join(DATA_DIR, "processed")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

# Chunking parameters
MAX_CHUNK_SIZE = 512  # Maximum number of tokens in a chunk
OVERLAP_SIZE = 50     # Number of tokens to overlap between chunks

def clean_text(text):
    """Clean and normalize text."""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep important punctuation
    text = re.sub(r'[^\w\s\.\,\;\:\(\)\[\]\-\"\'\?]', '', text)
    
    # Normalize whitespace
    text = text.strip()
    
    return text

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    logger.info(f"Extracting text from PDF: {pdf_path}")
    
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n\n"
        
        return text
    except Exception as e:
        logger.error(f"Error extracting text from PDF {pdf_path}: {e}")
        return ""

def chunk_text(text, max_chunk_size=MAX_CHUNK_SIZE, overlap_size=OVERLAP_SIZE):
    """Split text into overlapping chunks."""
    # Simple word-based chunking
    words = text.split()
    chunks = []
    
    if len(words) <= max_chunk_size:
        return [text]
    
    i = 0
    while i < len(words):
        chunk = ' '.join(words[i:i + max_chunk_size])
        chunks.append(chunk)
        i += max_chunk_size - overlap_size
    
    return chunks

def process_file(file_path, source_type):
    """Process a single file into chunks with metadata."""
    logger.info(f"Processing {file_path}...")
    
    try:
        # Check file extension
        if file_path.lower().endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        
        # Clean text
        cleaned_text = clean_text(text)
        
        # Extract filename without extension for document ID
        doc_id = os.path.splitext(os.path.basename(file_path))[0]
        
        # Split into chunks
        chunks = chunk_text(cleaned_text)
        
        # Create chunk objects with metadata
        chunk_objects = []
        for i, chunk in enumerate(chunks):
            chunk_object = {
                "chunk_id": f"{doc_id}_{i}",
                "doc_id": doc_id,
                "source_type": source_type,
                "source_file": file_path,
                "chunk_index": i,
                "text": chunk
            }
            chunk_objects.append(chunk_object)
        
        return chunk_objects
    
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        return []

def generate_embeddings(chunks, model_name="all-MiniLM-L6-v2"):
    """Generate embeddings for text chunks using SentenceTransformer."""
    logger.info(f"Generating embeddings using {model_name}...")
    
    # Load model
    model = SentenceTransformer(model_name)
    
    # Extract texts for embedding
    texts = [chunk["text"] for chunk in chunks]
    
    # Generate embeddings in batches
    batch_size = 32
    embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = model.encode(batch_texts, convert_to_tensor=True)
        embeddings.append(batch_embeddings)
    
    # Concatenate all embeddings
    if len(embeddings) > 0:
        all_embeddings = torch.cat(embeddings)
        return all_embeddings.cpu().numpy()
    else:
        return np.array([])

def process_all_documents():
    """Process all legal documents and generate embeddings."""
    all_chunks = []
    
    # Process CFR Title 8
    cfr_files = glob.glob(os.path.join(CFR_DIR, "*.*"))
    for file_path in cfr_files:
        if not os.path.basename(file_path).startswith("structure"):  # Skip structure files
            chunks = process_file(file_path, "CFR_Title_8")
            all_chunks.extend(chunks)
    
    # Process INA
    ina_files = glob.glob(os.path.join(INA_DIR, "*.*"))
    for file_path in ina_files:
        if not os.path.basename(file_path).startswith("structure"):
            chunks = process_file(file_path, "INA")
            all_chunks.extend(chunks)
    
    # Process USCIS Policy Manual
    policy_files = glob.glob(os.path.join(POLICY_DIR, "*.*"))
    for file_path in policy_files:
        if not os.path.basename(file_path).startswith("structure"):
            chunks = process_file(file_path, "USCIS_Policy")
            all_chunks.extend(chunks)
    
    logger.info(f"Total chunks created: {len(all_chunks)}")
    
    # Save chunks to JSON
    chunks_file = os.path.join(OUTPUT_DIR, "all_chunks.json")
    with open(chunks_file, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, indent=2)
    
    # Create DataFrame for easier handling
    df = pd.DataFrame(all_chunks)
    df.to_csv(os.path.join(OUTPUT_DIR, "all_chunks.csv"), index=False)
    
    # Generate embeddings
    embeddings = generate_embeddings(all_chunks)
    
    # Save embeddings
    embeddings_file = os.path.join(EMBEDDINGS_DIR, "chunk_embeddings.npy")
    np.save(embeddings_file, embeddings)
    
    # Save mapping between embeddings and chunks
    mapping = {i: chunk["chunk_id"] for i, chunk in enumerate(all_chunks)}
    with open(os.path.join(EMBEDDINGS_DIR, "embedding_mapping.pkl"), 'wb') as f:
        pickle.dump(mapping, f)
    
    logger.info(f"Preprocessing complete. Chunks saved to {chunks_file}")
    logger.info(f"Embeddings saved to {embeddings_file}")
    
    return len(all_chunks), embeddings.shape

if __name__ == "__main__":
    num_chunks, embedding_shape = process_all_documents()
    logger.info(f"Created {num_chunks} chunks with embedding shape {embedding_shape}")
