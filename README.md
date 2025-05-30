# Create README for backend
cat > README.md << 'EOF'
# Legal Search AI - Backend API

🤖 AI-powered legal search backend with RAG (Retrieval-Augmented Generation) for US Immigration Law.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python app.py
```

The API will be available at `http://localhost:8081`

## 📋 API Endpoints

- **Health Check**: `GET /api/health`
- **Search**: `POST /api/answer`

### Example Request
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"query":"What are H1B requirements?","use_llm":false}' \
  http://localhost:8081/api/answer
```

## 🛠️ Tech Stack

- **Framework**: Flask
- **AI/ML**: sentence-transformers, scikit-learn
- **Data**: 4,824 legal document chunks
- **Embeddings**: 384-dimensional vectors

## 🌐 Deployment

Deploy on [Render.com](https://render.com) using the included `render.yaml` configuration.

## 📄 License

Educational use only. Consult qualified immigration attorneys for legal advice.
EOF