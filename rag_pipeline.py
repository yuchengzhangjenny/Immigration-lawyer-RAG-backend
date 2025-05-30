"""
RAG Pipeline implementation for the Legal Search AI application.
This script implements the retrieval and generation components for answering
questions about US immigration law.
"""
import time
import os
import json
import numpy as np
import pickle
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import logging
import psutil

# Configure logging
logger = logging.getLogger("rag_pipeline")

# Directories - using relative paths
DATA_DIR = "data"
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")

# Model parameters
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5  # Number of chunks to retrieve
LLM_MODEL = "TheBloke/Llama-2-7B-Chat-GPTQ"  # 7B model with quantization for lower memory usage

class LegalSearchRAG:
    """
    Retrieval Augmented Generation system for legal search.
    """
    
    def __init__(self):
        """Initialize the RAG system."""
        self.model = None
        self.chunks_df = None
        self.embeddings = None
        self.embedding_mapping = None
        self.llm = None
        self.llm_tokenizer = None
        self.loaded = False
        self.llm_loaded = False
    
    def load_resources(self):
        """Load all necessary resources for the RAG system."""
        print("Loading resources...")
        
        # Load chunks
        chunks_file = os.path.join(PROCESSED_DIR, "all_chunks.csv")
        if os.path.exists(chunks_file):
            self.chunks_df = pd.read_csv(chunks_file)
            print(f"Loaded {len(self.chunks_df)} chunks from {chunks_file}")
        else:
            raise FileNotFoundError(f"Chunks file not found: {chunks_file}")
        
        # Load embeddings
        embeddings_file = os.path.join(EMBEDDINGS_DIR, "chunk_embeddings.npy")
        if os.path.exists(embeddings_file):
            self.embeddings = np.load(embeddings_file)
            print(f"Loaded embeddings with shape {self.embeddings.shape} from {embeddings_file}")
        else:
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")
        
        # Load mapping
        mapping_file = os.path.join(EMBEDDINGS_DIR, "embedding_mapping.pkl")
        if os.path.exists(mapping_file):
            with open(mapping_file, 'rb') as f:
                self.embedding_mapping = pickle.load(f)
            print(f"Loaded embedding mapping with {len(self.embedding_mapping)} entries")
        else:
            raise FileNotFoundError(f"Mapping file not found: {mapping_file}")
        
        # Load model
        self.model = SentenceTransformer(MODEL_NAME)
        print(f"Loaded SentenceTransformer model: {MODEL_NAME}")
        
        self.loaded = True
        print("All resources loaded successfully")
    
    def load_llm(self):
        """Load the LLM model for generation."""
        if not self.llm_loaded:
            print(f"Loading LLM model: {LLM_MODEL}")
            logger.info(f"Starting to load LLM model: {LLM_MODEL}")
            try:
                # Check if MPS (Metal Performance Shaders) is available on Mac
                if torch.backends.mps.is_available():
                    device = "mps"
                    logger.info("Using MPS (Apple Metal) for acceleration")
                elif torch.cuda.is_available():
                    device = "cuda"
                    device_info = torch.cuda.get_device_properties(0)
                    total_memory = device_info.total_memory / (1024 ** 3)  # Convert to GB
                    logger.info(f"GPU detected: {torch.cuda.get_device_name(0)} with {total_memory:.2f} GB total memory")
                else:
                    device = "cpu"
                    logger.info("No GPU detected, using CPU for inference")
                
                # Check available RAM
                ram_gb = psutil.virtual_memory().total / (1024 ** 3)
                logger.info(f"System has {ram_gb:.2f} GB RAM")
                
                # Install required libraries for quantized models
                try:
                    import optimum
                    import auto_gptq
                except ImportError:
                    logger.info("Installing optimum and auto-gptq packages...")
                    import subprocess
                    subprocess.check_call(["pip", "install", "optimum", "auto-gptq"])
                    logger.info("Installation complete")
                
                # Load tokenizer
                logger.info("Loading tokenizer...")
                self.llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
                
                # Import after installation to ensure it's available
                from optimum.gptq import GPTQConfig
                from auto_gptq import AutoGPTQForCausalLM
                
                # Load quantized model with memory optimizations
                logger.info(f"Loading 4-bit quantized model...")
                quantization_config = GPTQConfig(bits=4, group_size=128, desc_act=True)
                
                model = AutoGPTQForCausalLM.from_quantized(
                    LLM_MODEL,
                    quantization_config=quantization_config,
                    device=device,
                    use_safetensors=True,
                    trust_remote_code=True
                )
                
                # Create pipeline
                self.llm = pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=self.llm_tokenizer,
                    device_map=device
                )
                self.llm_loaded = True
                logger.info("LLM model loaded successfully")
                print("LLM model loaded successfully")
            except Exception as e:
                error_msg = f"Error loading LLM model: {str(e)}"
                logger.error(error_msg)
                logger.exception("Full traceback:")
                print(error_msg)
    
    def encode_query(self, query):
        """Encode a query into an embedding vector."""
        if not self.loaded:
            self.load_resources()
        
        return self.model.encode(query, convert_to_tensor=True).cpu().numpy()
    
    def retrieve(self, query, top_k=TOP_K):
        """Retrieve the most relevant chunks for a query."""
        if not self.loaded:
            self.load_resources()
        
        # Encode query
        query_embedding = self.encode_query(query)
        
        # Calculate similarity
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        # Get top-k indices
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        # Get corresponding chunks
        results = []
        for idx in top_indices:
            chunk_id = self.embedding_mapping[idx]
            chunk_row = self.chunks_df[self.chunks_df['chunk_id'] == chunk_id]
            
            if not chunk_row.empty:
                result = {
                    'chunk_id': chunk_id,
                    'doc_id': chunk_row['doc_id'].values[0],
                    'source_type': chunk_row['source_type'].values[0],
                    'source_file': chunk_row['source_file'].values[0],
                    'text': chunk_row['text'].values[0],
                    'similarity': float(similarities[idx])  # Convert numpy.float32 to Python float
                }
                results.append(result)
        
        return results
    
    def generate_answer_with_llm(self, query, retrieved_chunks):
        """
        Generate an answer based on retrieved chunks using DeepSeek LLM.
        """
        if not retrieved_chunks:
            return "I couldn't find relevant information to answer your question."
        
        # Try to load LLM if not already loaded
        if not self.llm_loaded:
            try:
                logger.info("Attempting to load the LLM model")
                self.load_llm()
            except Exception as e:
                logger.error(f"Failed to load LLM model: {e}")
                error_msg = f"[LLM LOADING FAILED: The DeepSeek model could not be loaded due to: {str(e)}]\n\n"
                return error_msg + self.generate_answer_default(query, retrieved_chunks)
        
        # If LLM loading failed, mention it and fall back to default
        if not self.llm_loaded:
            error_msg = "[LLM LOADING FAILED: The DeepSeek model could not be loaded, possibly due to memory constraints or model availability issues. Falling back to basic retrieval.]\n\n"
            return error_msg + self.generate_answer_default(query, retrieved_chunks)
        
        try:
            # Sort chunks by similarity
            sorted_chunks = sorted(retrieved_chunks, key=lambda x: x['similarity'], reverse=True)
            
            # Create context from chunks
            context = "\n\n".join([f"Source ({chunk['source_type']}): {chunk['text']}" for chunk in sorted_chunks])
            
            # Create citations
            citations = []
            for i, chunk in enumerate(sorted_chunks):
                source_name = os.path.basename(chunk['source_file'])
                citations.append(f"[{i+1}] {chunk['source_type']} - {source_name}")
            
            logger.info(f"Generating LLM response at timestamp: {time.time()}")
            # Create prompt
            prompt = f"""You are a legal assistant specializing in US immigration law. You have conducted some research and found primary information to support your advice. Now answer the question using your research:
Question: {query}

Context:
{context}

Answer:"""
            
            # Generate answer
            response = self.llm(
                prompt,
                max_new_tokens=512,
                num_return_sequences=1,
                temperature=0.3,
                do_sample=True,
                truncation=True
            )
            logger.info(f"Done Generating LLM response at timestamp: {time.time()}")
            
            # Extract answer text
            answer_text = response[0]["generated_text"]
            logger.info(f"Raw LLM response: {answer_text[:100]}...")
            
            # Clean up the answer (remove the prompt)
            if "Answer:" in answer_text:
                answer_text = answer_text.split("Answer:")[1].strip()
            else:
                logger.warning("Could not find 'Answer:' marker in LLM response")
                
            # Add citations to answer
            answer_text += "\n\nSources:\n" + "\n".join(citations)

                    # Sort chunks by similarity
            sorted_chunks = sorted(retrieved_chunks, key=lambda x: x['similarity'], reverse=True)
            
            # Create context from chunks
            context = "\n\n".join([f"Source ({chunk['source_type']}): {chunk['text']}" for chunk in sorted_chunks])
            
            # Create citations
            citations = []
            for i, chunk in enumerate(sorted_chunks):
                source_name = os.path.basename(chunk['source_file'])
                citations.append(f"[{i+1}] {chunk['source_type']} - {source_name}")
            
            # Create a structured answer
            answer = f"Based on the US immigration laws and regulations, here's information related to your query:\n\n"
            
            # Add the most relevant chunk as the primary answer
            answer += f"{sorted_chunks[0]['text']}\n\n"
            
            # Add additional information from other chunks
            if len(sorted_chunks) > 1:
                answer += "Additional relevant information:\n\n"
                for i, chunk in enumerate(sorted_chunks[1:], 1):
                    answer += f"- {chunk['text'][:200]}...\n\n"
            
            # Add citations
            answer += "Sources:\n"
            for citation in citations:
                answer += f"{citation}\n"
            
            return answer_text
            
        except Exception as e:
            logger.error(f"Error generating answer with LLM: {e}")
            print(f"Error with LLM generation: {e}")
            # Add error message before falling back
            error_msg = f"[LLM GENERATION FAILED: {str(e)}]\n\n"
            return error_msg + self.generate_answer_default(query, retrieved_chunks)
    
    def generate_answer_default(self, query, retrieved_chunks):
        """
        Generate an answer based on retrieved chunks without using an LLM.
        This is the original implementation and serves as a fallback.
        """
        if not retrieved_chunks:
            return "I couldn't find relevant information to answer your question."
        
        # Sort chunks by similarity
        sorted_chunks = sorted(retrieved_chunks, key=lambda x: x['similarity'], reverse=True)
        
        # Create context from chunks
        context = "\n\n".join([f"Source ({chunk['source_type']}): {chunk['text']}" for chunk in sorted_chunks])
        
        # Create citations
        citations = []
        for i, chunk in enumerate(sorted_chunks):
            source_name = os.path.basename(chunk['source_file'])
            citations.append(f"[{i+1}] {chunk['source_type']} - {source_name}")
        
        # Create a structured answer
        answer = f"Based on the US immigration laws and regulations, here's information related to your query:\n\n"
        
        # Add the most relevant chunk as the primary answer
        answer += f"{sorted_chunks[0]['text']}\n\n"
        
        # Add additional information from other chunks
        if len(sorted_chunks) > 1:
            answer += "Additional relevant information:\n\n"
            for i, chunk in enumerate(sorted_chunks[1:], 1):
                answer += f"- {chunk['text'][:200]}...\n\n"
        
        # Add citations
        answer += "Sources:\n"
        for citation in citations:
            answer += f"{citation}\n"
        
        return answer
    
    def answer_question(self, query, top_k=TOP_K, use_llm=False):
        """Answer a question using the RAG pipeline."""
        # Retrieve relevant chunks
        retrieved_chunks = self.retrieve(query, top_k)
        
        # Generate answer
        if use_llm:
            answer = self.generate_answer_with_llm(query, retrieved_chunks)
        else:
            answer = self.generate_answer_default(query, retrieved_chunks)
        
        return {
            'query': query,
            'answer': answer,
            'retrieved_chunks': retrieved_chunks
        }

# Example usage
if __name__ == "__main__":
    rag = LegalSearchRAG()
    rag.load_resources()
    
    # Example query
    query = "What are the requirements for a green card through marriage?"
    result = rag.answer_question(query, use_llm=True)
    
    print(f"Query: {result['query']}")
    print("\nAnswer:")
    print(result['answer'])
