"""
User interface for the Legal Search AI application.
This script implements a Flask web application that allows users to ask
questions about US immigration law and receive answers from the RAG system.
"""

import os
from flask import Flask, request, jsonify, render_template
import logging
from flask_cors import CORS

# Import the RAG pipeline
from rag_pipeline import LegalSearchRAG

# Create data directory if it doesn't exist
os.makedirs("data", exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data/app_log.txt"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("legal_search_app")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize RAG system
rag = LegalSearchRAG()

# Example questions to help users get started
EXAMPLE_QUESTIONS = [
    "What are the requirements for H1B visa?",
    "How long can I stay in the US with a tourist visa?",
    "What is the process for applying for asylum in the United States?",
    "What documents are needed for naturalization?",
    "What are the eligibility criteria for DACA?"
]

@app.route('/')
def home():
    return render_template('index.html', example_questions=EXAMPLE_QUESTIONS)

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    query = data.get('query', '')
    use_llm = data.get('use_llm', False)
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    try:
        result = rag.answer_question(query, use_llm=use_llm)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/answer', methods=['POST'])
def answer():
    """Endpoint for Webflow integration - maps to the same functionality as /search."""
    data = request.json
    query = data.get('query', '')
    use_llm = data.get('use_llm', False)
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    try:
        result = rag.answer_question(query, use_llm=use_llm)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok'})

def create_app():
    """Create and configure the Flask app."""
    # Ensure the templates directory exists
    os.makedirs('templates', exist_ok=True)
    
    # Ensure the static directory exists
    os.makedirs('static', exist_ok=True)
    
    # Load RAG resources
    try:
        rag.load_resources()
        logger.info("RAG resources loaded successfully")
    except Exception as e:
        logger.error(f"Error loading RAG resources: {str(e)}")
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=8081, debug=True)
