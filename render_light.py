#!/usr/bin/env python3
"""
Lightweight version for Render deployment
Reduces memory usage and startup time
"""

import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_light_app():
    """Create a lightweight Flask app for Render"""
    app = Flask(__name__)
    CORS(app)
    
    # Global variable for lazy loading
    rag_pipeline = None
    
    def get_rag_pipeline():
        """Lazy load the RAG pipeline only when needed"""
        global rag_pipeline
        if rag_pipeline is None:
            try:
                logger.info("Loading RAG pipeline...")
                # Try to import and initialize RAG
                from rag_pipeline import RAGPipeline
                rag_pipeline = RAGPipeline()
                logger.info("RAG pipeline loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load RAG pipeline: {e}")
                # Fallback to simple text search
                rag_pipeline = "fallback"
        return rag_pipeline
    
    @app.route('/')
    def api_info():
        """API information endpoint."""
        return jsonify({
            'name': 'Immigration Lawyer RAG Backend (Light)',
            'version': '1.0',
            'status': 'running',
            'mode': 'lightweight',
            'endpoints': {
                'health': '/api/health',
                'search': '/search (POST)',
                'test': '/test'
            }
        })
    
    @app.route('/api/health')
    def health():
        """Health check endpoint."""
        return jsonify({'status': 'ok', 'mode': 'lightweight'})
    
    @app.route('/test', methods=['GET', 'POST'])
    def test():
        """Simple test endpoint"""
        return jsonify({
            'status': 'success',
            'message': 'Lightweight version working!',
            'timestamp': str(datetime.now()),
            'mode': 'render-optimized'
        })
    
    @app.route('/search', methods=['POST'])
    def search():
        """Search endpoint with fallback functionality"""
        data = request.json
        query = data.get('query', '')
        use_llm = data.get('use_llm', False)
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        try:
            # Try to get RAG pipeline
            rag = get_rag_pipeline()
            
            if rag == "fallback":
                # Simple fallback response
                return jsonify({
                    'answer': f"This is a lightweight response for: {query}. Full RAG capabilities require more memory than available on this deployment.",
                    'query': query,
                    'chunks': [],
                    'mode': 'fallback',
                    'message': 'Upgrade deployment for full AI capabilities'
                })
            else:
                # Use full RAG pipeline
                result = rag.answer_question(query, use_llm=use_llm)
                result['mode'] = 'full'
                return jsonify(result)
                
        except Exception as e:
            logger.error(f"Error in search: {str(e)}")
            return jsonify({
                'error': 'Search temporarily unavailable',
                'query': query,
                'mode': 'error',
                'message': str(e)
            }), 500
    
    return app

if __name__ == '__main__':
    app = create_light_app()
    port = int(os.environ.get('PORT', 8081))
    
    logger.info(f"Starting lightweight app on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False) 