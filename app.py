"""
User interface for the Legal Search AI application.
This script implements a Flask web application that allows users to ask
questions about US immigration law and receive answers from the RAG system.
"""

import os
from flask import Flask, request, jsonify, render_template
import logging
from flask_cors import CORS
from datetime import datetime

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

# Initialize RAG system at startup (eager loading)
rag = LegalSearchRAG()
print("🚀 Loading RAG pipeline at startup...")
try:
    rag.load_resources()
    print("✅ RAG pipeline loaded successfully!")
    print(f"📊 Loaded {len(rag.chunks_df)} document chunks")
except Exception as e:
    print(f"❌ Failed to load RAG pipeline: {e}")
    rag = None

# Example questions to help users get started
EXAMPLE_QUESTIONS = [
    "What are the requirements for H1B visa?",
    "How long can I stay in the US with a tourist visa?",
    "What is the process for applying for asylum in the United States?",
    "What documents are needed for naturalization?",
    "What are the eligibility criteria for DACA?"
]

@app.route('/')
def api_info():
    """API information endpoint."""
    return jsonify({
        'name': 'Immigration Lawyer RAG Backend',
        'version': '1.0',
        'endpoints': {
            'health': '/api/health',
            'search': '/search (POST)',
            'test': '/test'
        },
        'status': 'running'
    })

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    query = data.get('query', '')
    use_llm = data.get('use_llm', False)
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    # Check if RAG pipeline loaded successfully
    if rag is None:
        return jsonify({
            'error': 'RAG pipeline not available',
            'message': 'The AI models failed to load at startup. Please try again later.',
            'query': query
        }), 503
    
    try:
        result = rag.answer_question(query, use_llm=use_llm)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/test1', methods=['POST'])
def test_search():
    """Test endpoint with predefined query"""
    data = request.json
    query = "What are the requirements for a green card through marriage?"
    use_llm = data.get('use_llm', False)
    
    # Check if RAG pipeline loaded successfully
    if rag is None:
        return jsonify({
            'error': 'RAG pipeline not available',
            'message': 'The AI models failed to load at startup. Please try again later.',
            'query': query
        }), 503
    
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

@app.route('/test', methods=['GET', 'POST'])
def test_endpoint():
    """Simple test endpoint that doesn't use RAG."""
    return jsonify({
        'status': 'success',
        'message': 'Basic Flask functionality working',
        'method': request.method,
        'timestamp': str(datetime.now())
    })

@app.route('/simple-test', methods=['GET', 'POST'])
def simple_test():
    """Simple test endpoint without RAG pipeline."""
    return jsonify({
        'status': 'success',
        'message': 'Simple test working!',
        'environment': 'production' if os.environ.get('PORT') else 'development',
        'timestamp': str(datetime.now())
    })

@app.route('/api/status')
def startup_status():
    """Check if RAG pipeline is ready."""
    if rag is None:
        return jsonify({
            'status': 'loading',
            'message': 'AI models are still loading, please wait...',
            'ready': False
        })
    elif not rag.loaded:
        return jsonify({
            'status': 'loading', 
            'message': 'RAG pipeline initializing...',
            'ready': False
        })
    else:
        return jsonify({
            'status': 'ready',
            'message': 'All systems ready!',
            'ready': True,
            'chunks_loaded': len(rag.chunks_df) if rag.chunks_df is not None else 0
        })

def create_app():
    """Create and configure the Flask app."""
    # Ensure the templates directory exists
    os.makedirs('templates', exist_ok=True)
    
    # Ensure the static directory exists
    os.makedirs('static', exist_ok=True)
    
    # RAG resources already loaded at startup - no need to reload here
    logger.info("App created successfully")
    
    return app

if __name__ == '__main__':
    app = create_app()
    port = int(os.environ.get('PORT', 8081))
    app.run(host='0.0.0.0', port=port, debug=False)
