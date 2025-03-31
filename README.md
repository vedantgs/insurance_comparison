# Multi-Agent Experimentation Project

## Overview
This project contains experimental code for multi-agent systems using various language models and frameworks.

## Requirements
The project requires several Python libraries:
- streamlit - For building interactive web applications
- langchain and related packages - For building LLM-powered applications
- FAISS - For vector similarity search and storage
- PDF processing libraries - For handling document extraction

## Setup
1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
streamlit run app.py
```

## Features
### Document Processing and Knowledge Base
- PDF document ingestion and parsing using Unstructured and PyPDF2
- Text chunking and preprocessing for optimal embedding
- Semantic search capabilities over document corpus

### Language Model Integration
- Multi-model support with Groq and Google Generative AI models
- Model switching capability for comparing responses
- Optimized prompting strategies for different use cases

### Vector Storage and Retrieval
- FAISS-based vector database for efficient similarity search
- Document retrieval based on semantic relevance
- Customizable retrieval parameters (k-nearest neighbors, similarity thresholds)

### Interactive User Interface
- Streamlit-based web interface for easy interaction
- Real-time query processing and response generation
- Document upload and management capabilities

### Multi-Agent Architecture
- Agent collaboration frameworks for complex tasks
- Role-based agent specialization
- Memory and context management between agent interactions

## Project Structure
- `requirements.txt` - Contains all required Python dependencies
- Additional files for application logic and utilities
