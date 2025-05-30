"""
Validation script for the Legal Search AI application.
This script tests the RAG pipeline with sample queries and validates
the answers against source documents.
"""

import os
import sys
import json
import logging
from tqdm import tqdm

# Add the project root to the path
sys.path.append('/home/ubuntu/legal_search_ai')

# Import the RAG pipeline
from src.rag_pipeline import LegalSearchRAG

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("/home/ubuntu/legal_search_ai/data/validation_log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("validation")

# Sample test queries covering different aspects of immigration law
TEST_QUERIES = [
    "What are the requirements for a green card through marriage?",
    "How do I apply for asylum in the United States?",
    "What is the difference between an immigrant visa and a nonimmigrant visa?",
    "What are the eligibility requirements for naturalization?",
    "How long can I stay in the US with a B1/B2 visa?",
    "What is the USCIS policy on DACA?",
    "What happens if my visa expires while I'm in the US?",
    "What are the requirements for an H-1B visa?",
    "How can I sponsor a family member for immigration?",
    "What is the process for citizenship through naturalization?"
]

def validate_rag_pipeline():
    """Validate the RAG pipeline with test queries."""
    logger.info("Starting validation of RAG pipeline")
    
    # Initialize RAG system
    rag = LegalSearchRAG()
    
    try:
        # Load resources
        rag.load_resources()
        logger.info("RAG resources loaded successfully")
        
        # Create results directory
        results_dir = "/home/ubuntu/legal_search_ai/data/validation_results"
        os.makedirs(results_dir, exist_ok=True)
        
        # Process each test query
        all_results = []
        
        for query in tqdm(TEST_QUERIES, desc="Processing test queries"):
            logger.info(f"Testing query: {query}")
            
            # Get answer
            result = rag.answer_question(query)
            
            # Add to results
            all_results.append({
                'query': query,
                'answer': result['answer'],
                'sources': [
                    {
                        'chunk_id': chunk['chunk_id'],
                        'source_type': chunk['source_type'],
                        'source_file': os.path.basename(chunk['source_file']),
                        'similarity': float(chunk['similarity']),
                        'text_snippet': chunk['text'][:200] + "..." if len(chunk['text']) > 200 else chunk['text']
                    }
                    for chunk in result['retrieved_chunks']
                ]
            })
            
            # Save individual result
            query_filename = query.lower().replace('?', '').replace(' ', '_')[:50] + '.json'
            with open(os.path.join(results_dir, query_filename), 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, default=str)
        
        # Save all results
        with open(os.path.join(results_dir, 'all_validation_results.json'), 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        logger.info(f"Validation complete. Results saved to {results_dir}")
        
        # Generate validation report
        generate_validation_report(all_results, results_dir)
        
        return all_results
    
    except Exception as e:
        logger.error(f"Error during validation: {str(e)}")
        raise

def generate_validation_report(results, results_dir):
    """Generate a human-readable validation report."""
    report_path = os.path.join(results_dir, 'validation_report.md')
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Legal Search AI Validation Report\n\n")
        f.write("This report contains the results of validating the RAG pipeline with sample queries.\n\n")
        
        for i, result in enumerate(results, 1):
            f.write(f"## Query {i}: {result['query']}\n\n")
            
            # Write answer
            f.write("### Answer:\n\n")
            f.write(f"{result['answer']}\n\n")
            
            # Write sources
            f.write("### Sources:\n\n")
            for j, source in enumerate(result['sources'], 1):
                f.write(f"**Source {j}:** {source['source_type']} - {source['source_file']} (Similarity: {source['similarity']:.2f})\n\n")
                f.write(f"Snippet: {source['text_snippet']}\n\n")
            
            f.write("---\n\n")
    
    logger.info(f"Validation report generated at {report_path}")
    return report_path

if __name__ == "__main__":
    validate_rag_pipeline()
