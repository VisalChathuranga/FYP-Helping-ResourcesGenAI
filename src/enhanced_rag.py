# src/enhanced_rag.py
from typing import Dict, Any, List
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.agents import SimpleAgentOrchestrator
from src.helper import get_gemini_embeddings
import os

class EnhancedRAGSystem:
    """Enhanced RAG system with AI agents and web search"""
    
    def __init__(self):
        self.setup_llm()
        self.setup_vectorstore()
        self.setup_agents()
        self.setup_chains()
    
    def setup_llm(self):
        """Initialize the language model"""
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            temperature=0.7,
            max_output_tokens=8000,
        )
    
    def setup_vectorstore(self):
        """Setup FAISS vectorstore"""
        try:
            embeddings = get_gemini_embeddings()
            self.docsearch = FAISS.load_local(
                folder_path="faiss_new_index",
                embeddings=embeddings,
                allow_dangerous_deserialization=True
            )
            self.retriever = self.docsearch.as_retriever(
                search_type="similarity", 
                search_kwargs={"k": 7}
            )
            print("✅ FAISS vectorstore loaded successfully")
        except Exception as e:
            print(f"⚠️  Warning: Could not load FAISS vectorstore: {e}")
            self.docsearch = None
            self.retriever = None
    
    def setup_agents(self):
        """Setup AI agents"""
        self.agent_orchestrator = SimpleAgentOrchestrator()
        print("✅ AI agents initialized successfully")
    
    def setup_chains(self):
        """Setup LangChain chains"""
        system_prompt = """You are an advanced AI coding assistant with access to:
1. A comprehensive knowledge base of programming documentation
2. Real-time web search capabilities
3. Code execution and validation tools
4. Multi-agent collaboration system

Your capabilities include:
- Generating production-ready code with comprehensive error handling
- Searching the web for the latest programming trends and solutions
- Validating and testing code snippets
- Providing architectural guidance and best practices
- Debugging and optimizing existing code
- Creating complete project structures and configurations

When responding:
1. Use the provided context from the knowledge base
2. Leverage web search for current information and best practices
3. Generate complete, working solutions with proper documentation
4. Include error handling and edge case considerations
5. Provide testing examples and usage instructions
6. Suggest improvements and optimizations

Context from knowledge base:
{context}

Remember to be thorough, accurate, and provide production-ready solutions.
"""
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        if self.retriever:
            self.question_answer_chain = create_stuff_documents_chain(self.llm, self.prompt)
            self.rag_chain = create_retrieval_chain(self.retriever, self.question_answer_chain)
            print("✅ RAG chains initialized successfully")
        else:
            print("⚠️  Warning: RAG chains not initialized due to missing vectorstore")
    
    def process_query(self, query: str, use_agents: bool = True) -> Dict[str, Any]:
        """Process a query using either traditional RAG or enhanced agent system"""
        
        if use_agents and self.should_use_agents(query):
            # Use advanced agent system
            print("🤖 Using advanced agent system...")
            
            # Get context from vector store if available
            context = ""
            docs = []
            if self.retriever:
                try:
                    docs = self.retriever.get_relevant_documents(query)
                    context = "\n\n".join([doc.page_content for doc in docs])
                except Exception as e:
                    print(f"Warning: Could not retrieve documents: {e}")
            
            # Process with agents
            agent_response = self.agent_orchestrator.process_query(query, context)
            
            return {
                "answer": agent_response,
                "source_documents": docs,
                "method": "agent_enhanced",
                "context": context
            }
        else:
            # Use traditional RAG or fallback to direct LLM
            print("📚 Using traditional RAG system...")
            
            if self.rag_chain:
                try:
                    response = self.rag_chain.invoke({"input": query})
                    return {
                        "answer": response["answer"],
                        "source_documents": response.get("context", []),
                        "method": "traditional_rag",
                        "context": response.get("context", "")
                    }
                except Exception as e:
                    print(f"RAG chain error: {e}")
            
            # Fallback to direct LLM
            print("💬 Falling back to direct LLM...")
            try:
                response = self.llm.invoke([
                    {"role": "system", "content": "You are a helpful AI coding assistant. Provide clear, practical solutions with code examples."},
                    {"role": "user", "content": query}
                ])
                return {
                    "answer": response.content,
                    "source_documents": [],
                    "method": "direct_llm",
                    "context": ""
                }
            except Exception as e:
                return {
                    "answer": f"I encountered an error processing your request: {str(e)}",
                    "source_documents": [],
                    "method": "error",
                    "context": ""
                }
    
    def should_use_agents(self, query: str) -> bool:
        """Determine if query should use advanced agent system"""
        agent_keywords = [
            "generate", "create", "build", "develop", "implement",
            "latest", "current", "new", "modern", "advanced",
            "best practices", "optimization", "performance",
            "architecture", "design pattern", "framework",
            "tutorial", "example", "demo", "project", "how to"
        ]
        
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in agent_keywords)
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        vectorstore_size = 0
        if self.docsearch:
            try:
                vectorstore_size = self.docsearch.index.ntotal
            except:
                vectorstore_size = "Unknown"
        
        return {
            "vectorstore_size": vectorstore_size,
            "retriever_k": self.retriever.search_kwargs["k"] if self.retriever else 0,
            "llm_model": "gemini-2.5-pro",
            "agents_enabled": True,
            "tools_available": ["web_search", "code_execution", "validation"],
            "vectorstore_available": self.docsearch is not None
        }