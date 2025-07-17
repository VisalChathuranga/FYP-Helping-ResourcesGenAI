from src.helper import load_pdf_file, text_split, get_gemini_embeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os

load_dotenv()

GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Load and process PDFs
data_dir = "Data"  # Update this path
extracted_data = load_pdf_file(data_dir)
text_chunks = text_split(extracted_data)
embeddings = get_gemini_embeddings()

# Create FAISS index from documents
docsearch = FAISS.from_documents(documents=text_chunks, embedding=embeddings)

# Save the FAISS index locally
docsearch.save_local("faiss_new_index")

# Load the FAISS index later (if needed)
docsearch = FAISS.load_local(
    folder_path="faiss_new_index",
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)