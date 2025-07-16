from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

import os
import sys
sys.path.append("D:\\Final Year Project\\Resources Gen AI\\src")
from PDFExtractor import PDFExtractor

# Extract Data From the PDF Files
def load_pdf_file(data_dir):
    extractor = PDFExtractor()
    output_dir = "extracted_results"
    extractor.process_directory(data_dir, output_dir)
    
    # Load extracted text files into Documents
    documents = []
    for root, _, files in os.walk(output_dir):
        for file in files:
            if file.endswith(".txt"):
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    content = f.read()
                    relative_path = os.path.relpath(os.path.join(root, file), output_dir)
                    documents.append(Document(page_content=content, metadata={"source": relative_path}))
    
    return documents

# Split the Data into Text Chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

def get_gemini_embeddings():
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", 
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    return embeddings