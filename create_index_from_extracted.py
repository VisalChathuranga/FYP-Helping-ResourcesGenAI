from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set Google API key
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Load existing extracted text files
def load_extracted_files(extracted_dir):
    documents = []
    for root, _, files in os.walk(extracted_dir):
        for file in files:
            if file.endswith(".txt"):
                with open(os.path.join(root, file), "r", encoding="utf-8") as f:
                    content = f.read()
                    relative_path = os.path.relpath(os.path.join(root, file), extracted_dir)
                    documents.append(Document(page_content=content, metadata={"source": relative_path}))
    return documents

# Split text into chunks
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

# Create and save FAISS index
def create_faiss_index():
    extracted_dir = "extracted_results"
    if not os.path.exists(extracted_dir):
        print(f"Error: {extracted_dir} directory not found. Please ensure PDF extraction has been run.")
        return

    # Load documents
    extracted_data = load_extracted_files(extracted_dir)
    if not extracted_data:
        print("Error: No text files found in extracted_results.")
        return

    # Split into chunks
    text_chunks = text_split(extracted_data)

    # Get embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )

    # Create FAISS index
    docsearch = FAISS.from_documents(documents=text_chunks, embedding=embeddings)

    # Save the index
    docsearch.save_local("faiss_new_index")
    print("FAISS index saved as faiss_new_index.")

if __name__ == "__main__":
    create_faiss_index()