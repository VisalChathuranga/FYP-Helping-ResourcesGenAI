# 🤖 CodeGenix: AI-Powered Coding Assistant Chatbot

![Python](https://img.shields.io/badge/Python-3.8%2B-blue) ![Flask](https://img.shields.io/badge/Flask-2.0%2B-green) ![LangChain](https://img.shields.io/badge/LangChain-0.2%2B-orange) ![License](https://img.shields.io/badge/License-MIT-blue)

**CodeGenix** is a cutting-edge web-based chatbot that transforms complex PDF documents into a powerful knowledge base, driving an intelligent coding assistant. Built with Flask, LangChain, FAISS, and Google Generative AI, it leverages advanced PDF extraction, Retrieval-Augmented Generation (RAG), AI agents for real-time web searches and code validation, and session-based conversation memory to deliver precise, context-aware coding solutions.

---

## 📑 Advanced PDF Extraction

CodeGenix excels with its state-of-the-art PDF extraction pipeline, powered by the `EnhancedPDFExtractor` class, designed to process complex documents with unparalleled accuracy and efficiency:

- **Multi-Strategy Extraction**:
  - Utilizes `UnstructuredFileLoader` for high-resolution native text extraction.
  - Falls back to `PyPDFium2Loader` for enhanced table and structured data extraction.
  - Employs OCR (Tesseract and Poppler) for scanned or image-based PDFs, ensuring comprehensive content capture.
- **Quality Assurance**:
  - Validates text quality using metrics like character distribution and OCR artifact detection, triggering OCR if quality is below threshold (configurable via `ExtractorConfig`).
  - Preserves metadata (e.g., source file, page numbers, extraction method) for traceability.
- **Efficient Indexing**:
  - Splits text into 500-character chunks with 20-character overlap using `RecursiveCharacterTextSplitter`.
  - Generates embeddings with Google’s `embedding-001` model, stored in a FAISS vector index for rapid retrieval.
- **Enhanced Features**:
  - **Parallel Processing**: Uses `ThreadPoolExecutor` for concurrent PDF processing, configurable via `max_workers`.
  - **Caching**: Stores processed file metadata in a JSON cache to skip redundant processing.
  - **Flexible Output**: Supports `txt`, `json`, and `markdown` output formats with customizable metadata inclusion.
  - **Retry Logic**: Implements automatic retries for failed extractions with configurable delays.
  - **Metrics Tracking**: Records processing statistics (success rate, pages processed, OCR usage) in `processing_metrics.json`.

This advanced pipeline enables CodeGenix to handle technical PDFs, research papers, and coding documentation, creating a robust knowledge base for the chatbot.

---

## 📹 Demo Video

Watch a demonstration of CodeGenix in action, showcasing its PDF-based query handling and coding assistance capabilities:

[View CodeGenix ChatBot Demo](https://youtu.be/PZloAqJAn84)

*Note*: Hosted on YouTube for better accessibility.

---

## 🚀 Key Features

- **RAG-Powered Retrieval**: Uses FAISS and LangChain to retrieve relevant documents from the PDF knowledge base for accurate responses.
- **AI Agents**:
  - **SearchAgent**: Conducts real-time web searches via DuckDuckGo for the latest coding trends and solutions.
  - **CodeGenerationAgent**: Produces production-ready code with error handling, documentation, and best practices.
  - **ValidationAgent**: Ensures code quality by checking syntax, logic, security, and adherence to standards.
- **Conversation Memory**: Retains up to 10 messages per session in Flask for context-aware interactions.
- **Web Interface**: Provides a user-friendly Flask-based frontend (`chat.html`) for seamless query submission.
- **Custom Tools**: Offers utilities for web search, code execution, package installation, and code formatting.

---

## 🛠️ Project Structure

```
CodeGenix/
├── Data/                    # Input PDF files
├── extracted_results/       # Extracted text from PDFs
├── faiss_new_index/         # FAISS vector index
├── src/                     # Source code
│   ├── __init__.py          # Package initializer
│   ├── agents.py            # AI agents for search, code generation, and validation
│   ├── enhanced_rag.py      # RAG system with agent integration
│   ├── helper.py            # Utilities for PDF processing and embeddings
│   ├── PDFExtractor.py      # Advanced PDF text extraction with OCR
│   ├── prompt.py            # System prompt for LLM
│   ├── tools.py             # Custom tools for AI agents
├── .env                     # Environment variables (GOOGLE_API_KEY)
├── app.py                   # Flask web server
├── store_index.py           # PDF indexing script
├── template.py              # Project structure setup
├── requirements.txt         # Python dependencies
├── research/trials.ipynb    # Experimental Jupyter notebook
```

---

## 📋 Prerequisites

- **Python**: 3.8 or higher
- **Tesseract OCR**: For PDF text extraction ([Installation Guide](https://github.com/UB-Mannheim/tesseract/wiki))
- **Poppler**: For PDF-to-image conversion ([Installation Guide](https://blog.alivate.com.au/poppler-windows/))
- **Google API Key**: For Google Generative AI embeddings and LLM
- **Dependencies**: Listed in `requirements.txt`

---

## 🔧 Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/VisalChathuranga/CodeGenix--AI-Powered-Coding-Assistant-Chatbot.git
   cd CodeGenix
   ```

2. **Set Up Environment**:
   - Create a `.env` file in the root directory:
     ```plaintext
     GOOGLE_API_KEY=your_google_api_key
     ```
   - Install Tesseract and Poppler, and add them to your system PATH.

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare Knowledge Base**:
   - Place PDF files (e.g., coding tutorials, documentation) in the `Data` directory.
   - Run the indexing script to extract text and create the FAISS index:
     ```bash
     python store_index.py
     ```

5. **Run the Application**:
   - Start the Flask server:
     ```bash
     python app.py
     ```
   - Access the chatbot at `http://localhost:8080` in your browser.

---

## 🌟 Usage

1. **Interact with the Chatbot**:
   - Open `http://localhost:8080` to access the web interface (`chat.html`).
   - Submit queries related to your PDF knowledge base or coding tasks, e.g.:
     - "Summarize the algorithms in Data/algorithms.pdf."
     - "Write a Python function to implement merge sort."
     - "Optimize the previous function."
   - The chatbot leverages the indexed PDFs and conversation history for context-aware responses.

2. **Advanced Queries**:
   - Request code generation, tutorials, or best practices (e.g., "Build a Flask REST API with authentication").
   - AI agents perform web searches and validate code for accuracy and quality.

3. **Example Interaction**:
   ```
   User: Summarize the sorting algorithms in Data/algorithms.pdf.
   Bot: The PDF details bubble sort, merge sort, and quicksort with pseudocode and complexity analysis...
   User: Implement merge sort in Python.
   Bot: Here's an optimized Python merge sort implementation with error handling...
   User: Optimize it further.
   Bot: To enhance the merge sort, we can reduce memory usage by...
   ```

---

## 🧠 How It Works

1. **PDF Extraction and Indexing**:
   - `PDFExtractor.py` processes PDFs using multiple strategies (Unstructured, PyPDFium2, OCR), saving text to `extracted_results`.
   - `store_index.py` and `helper.py` split text into chunks, generate embeddings, and create a FAISS index.

2. **Query Processing**:
   - `app.py` serves the Flask web interface, handling queries via the `/get` endpoint.
   - `enhanced_rag.py` integrates RAG with AI agents (`agents.py`), selecting traditional RAG or agent-based processing based on query complexity.
   - `agents.py` uses `SearchAgent`, `CodeGenerationAgent`, and `ValidationAgent` with tools from `tools.py` for advanced tasks.

3. **Conversation History**:
   - `app.py` stores up to 10 messages in Flask sessions, passing history to the LLM for context-aware responses.

4. **Tools**:
   - `tools.py` provides utilities like web search (DuckDuckGo), code execution, package installation, and formatting.

---

## 📦 Dependencies

Key dependencies (see `requirements.txt` for full list):
- `flask`: Web server
- `langchain`: RAG and agent framework
- `langchain-community`: FAISS vector store
- `langchain-google-genai`: Google LLM and embeddings
- `pytesseract`, `pdf2image`: PDF processing
- `python-dotenv`: Environment variable management

---

## 🛡️ Limitations

- Conversation history is session-based and resets on page refresh.
- Requires internet access for Google API and web searches.
- Large PDF collections may increase FAISS index size; adjust chunk size in `helper.py` if needed.

---

## 🌱 Future Improvements

- **Persistent History**: Store chat history in a database (e.g., SQLite) for cross-session persistence.
- **Enhanced Frontend**: Improve `chat.html` to display chat history and add a "Clear History" button.
- **Agent Enhancements**: Integrate GitHub search or advanced linters in `tools.py`.
- **Performance**: Cache frequent queries and optimize FAISS index size.

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/YourFeature`).
3. Submit a pull request with clear descriptions.

---

## 📬 Contact

For questions or feedback, contact [e19056@eng.pdn.ac.lk](mailto:e19056@eng.pdn.ac.lk) or open an issue on GitHub.

---

*Built with 💻 and ☕ by [Visal_Chathuranga]*