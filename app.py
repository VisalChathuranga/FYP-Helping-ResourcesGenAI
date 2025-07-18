from flask import Flask, render_template, jsonify, request, session
from src.helper import get_gemini_embeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os
import markdown
import bleach
from markdown.extensions import codehilite

app = Flask(__name__)
app.secret_key = os.urandom(24)  # Required for session management

load_dotenv()

GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# Initialize memory for conversation history
memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="input",
    return_messages=True
)

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
    "You are an advanced AI assistant specializing in question-answering, code generation, and internet-based research. "
    "Utilize the provided context, conversation history, and collaborate with AI agents to search the internet for the latest information, frameworks, or libraries to generate precise, well-commented code snippets in the most suitable programming language, or provide detailed explanations if code is not applicable. "
    "Ensure code includes error handling, documentation, and best practices; if the answer or required data is unknown or context is insufficient, state 'I don't know'."
    "\n\n"
    "IMPORTANT FORMATTING RULES:"
    "- Use proper markdown formatting for all responses"
    "- Wrap code snippets in triple backticks with language specification (e.g., ```python)"
    "- Use **bold** for important terms and concepts"
    "- Use *italic* for emphasis"
    "- Use proper headings with # for structure"
    "- Use bullet points or numbered lists for step-by-step instructions"
    "- Ensure code is properly indented and formatted"
    "\n\n"
    "Context: {context}\n"
    "Chat History: {chat_history}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

def process_markdown(text):
    """Convert markdown to HTML with syntax highlighting"""
    try:
        # Configure markdown with code highlighting - FIXED VERSION
        md = markdown.Markdown(extensions=[
            'codehilite',
            'fenced_code',
            'tables',
            'nl2br'
        ], extension_configs={
            'codehilite': {
                'css_class': 'highlight',
                'use_pygments': True,
                'noclasses': True,
                'pygments_style': 'monokai'  # Changed from 'style' to 'pygments_style'
            }
        })
        
        # Convert markdown to HTML
        html = md.convert(text)
        
        # Sanitize HTML to prevent XSS (allow common tags)
        allowed_tags = [
            'p', 'br', 'strong', 'em', 'u', 'ol', 'ul', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
            'blockquote', 'code', 'pre', 'table', 'thead', 'tbody', 'tr', 'th', 'td',
            'div', 'span', 'a', 'img'
        ]
        
        allowed_attributes = {
            'div': ['class', 'style'],
            'span': ['class', 'style'],
            'pre': ['class', 'style'],
            'code': ['class', 'style'],
            'a': ['href', 'title'],
            'img': ['src', 'alt', 'title']
        }
        
        clean_html = bleach.clean(html, tags=allowed_tags, attributes=allowed_attributes)
        return clean_html
        
    except Exception as e:
        print(f"Error processing markdown: {e}")
        # Return formatted text as fallback
        return text.replace('\n', '<br>')

@app.route("/")
def index():
    # Clear session history when loading the main page
    session.pop('chat_history', None)
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    try:
        msg = request.form["msg"]
        print(f"User input: {msg}")

        # Load or initialize chat history from session
        chat_history = session.get('chat_history', [])
        
        # Update the input to include chat history context
        response = rag_chain.invoke({
            "input": msg,
            "chat_history": chat_history
        })
        
        # Process the response through markdown
        processed_response = process_markdown(response["answer"])
        
        # Update chat history
        chat_history.append({"human": msg, "ai": response["answer"]})
        session['chat_history'] = chat_history[-10:]  # Keep last 10 messages to manage memory
        
        print(f"Response: {response['answer']}")
        return processed_response
    
    except Exception as e:
        print(f"Error in chat: {str(e)}")
        return f"Error processing your request: {str(e)}"

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)