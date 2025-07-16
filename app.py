from flask import Flask, render_template, jsonify, request
from src.helper import get_gemini_embeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

app = Flask(__name__)

load_dotenv()

GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Load the saved FAISS index
embeddings = get_gemini_embeddings()
docsearch = FAISS.load_local(
    folder_path="faiss_new_index",
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 5})

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    temperature=0.8,
    max_output_tokens=50000,
)

system_prompt = (
    "You are an advanced AI assistant specializing in question-answering and code generation. "
    "Utilize the provided context to generate precise, well-commented code snippets in the most suitable programming language, or provide detailed explanations if code is not applicable. "
    "If the answer is unknown or context is insufficient, state 'I don’t know'; limit responses to three sentences and ensure conciseness."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)