# src/agents.py
from typing import List, Dict, Any, Optional
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import Tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import Graph, StateGraph, END
from langgraph.prebuilt import ToolExecutor
from pydantic import BaseModel, Field
import json
import re
import os
from datetime import datetime

class AgentState(BaseModel):
    """State object for the agent workflow"""
    query: str = ""
    context: str = ""
    search_results: List[Dict] = Field(default_factory=list)
    code_snippets: List[str] = Field(default_factory=list)
    final_answer: str = ""
    step: str = "start"
    error: Optional[str] = None

class SearchAgent:
    """Agent for web search and information gathering"""
    
    def __init__(self):
        self.search_tool = DuckDuckGoSearchRun()
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            temperature=0.3,
            max_output_tokens=4000,
        )
    
    def search_web(self, query: str, max_results: int = 5) -> List[Dict]:
        """Search the web for relevant information"""
        try:
            # Enhanced search queries for coding
            search_queries = [
                f"{query} tutorial examples code",
                f"{query} best practices implementation",
                f"{query} latest updates 2024 2025"
            ]
            
            results = []
            for search_query in search_queries:
                try:
                    result = self.search_tool.run(search_query)
                    results.append({
                        "query": search_query,
                        "content": result,
                        "timestamp": datetime.now().isoformat()
                    })
                except Exception as e:
                    print(f"Search error for '{search_query}': {e}")
                    continue
            
            return results[:max_results]
        except Exception as e:
            print(f"Web search failed: {e}")
            return []
    
    def extract_code_snippets(self, content: str) -> List[str]:
        """Extract code snippets from search results"""
        code_patterns = [
            r'```[\w]*\n(.*?)\n```',  # Markdown code blocks
            r'<code>(.*?)</code>',     # HTML code tags
            r'`([^`\n]+)`',            # Inline code
        ]
        
        snippets = []
        for pattern in code_patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            snippets.extend(matches)
        
        return [snippet.strip() for snippet in snippets if len(snippet.strip()) > 10]

class CodeGenerationAgent:
    """Agent for advanced code generation and optimization"""
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            temperature=0.7,
            max_output_tokens=8000,
        )
    
    def generate_code(self, query: str, context: str, search_results: List[Dict]) -> str:
        """Generate advanced code based on query, context, and search results"""
        
        # Combine search results
        search_content = "\n".join([result.get("content", "") for result in search_results])
        
        system_prompt = """You are an expert software engineer and AI coding assistant with deep expertise in:
- Full-stack development (Python, JavaScript, React, Node.js, etc.)
- AI/ML frameworks (TensorFlow, PyTorch, Langchain, etc.)
- Database design and optimization
- Cloud platforms and DevOps
- Software architecture and design patterns
- Latest programming trends and best practices

Your task is to generate high-quality, production-ready code that:
1. Follows best practices and coding standards
2. Includes comprehensive error handling
3. Has detailed comments and documentation
4. Is modular and maintainable
5. Uses the latest features and libraries
6. Includes testing examples where appropriate

If you need more information, clearly state what's missing and provide the best solution possible with the available information."""

        human_prompt = f"""
Query: {query}

Context from knowledge base:
{context}

Recent web search results:
{search_content}

Please generate a comprehensive solution that includes:
1. Complete, working code with proper imports
2. Detailed comments explaining the logic
3. Error handling and edge cases
4. Usage examples
5. Any additional setup or configuration needed

Make sure the code is production-ready and follows current best practices.
"""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=human_prompt)
            ]
            
            response = self.llm.invoke(messages)
            return response.content
            
        except Exception as e:
            return f"Error generating code: {str(e)}"

class ValidationAgent:
    """Agent for code validation and testing"""
    
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            temperature=0.2,
            max_output_tokens=4000,
        )
    
    def validate_code(self, code: str, query: str) -> Dict[str, Any]:
        """Validate generated code for syntax, logic, and best practices"""
        
        validation_prompt = f"""
Please analyze the following code and provide a comprehensive validation report:

Code to validate:
{code}

Original query: {query}

Please check for:
1. Syntax errors
2. Logic errors
3. Security vulnerabilities
4. Performance issues
5. Best practice violations
6. Missing imports or dependencies
7. Error handling completeness
8. Code readability and maintainability

Provide your response in JSON format with the following structure:
{{
    "is_valid": true/false,
    "syntax_errors": [],
    "logic_errors": [],
    "security_issues": [],
    "performance_issues": [],
    "best_practice_violations": [],
    "missing_dependencies": [],
    "suggestions": [],
    "overall_score": 0-10
}}
"""

        try:
            response = self.llm.invoke([HumanMessage(content=validation_prompt)])
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            else:
                return {"is_valid": False, "error": "Could not parse validation response"}
                
        except Exception as e:
            return {"is_valid": False, "error": f"Validation failed: {str(e)}"}

class SimpleAgentOrchestrator:
    """Simplified orchestrator without LangGraph dependency"""
    
    def __init__(self):
        self.search_agent = SearchAgent()
        self.code_agent = CodeGenerationAgent()
        self.validation_agent = ValidationAgent()
    
    def process_query(self, query: str, context: str = "") -> str:
        """Process a query through the agent workflow"""
        try:
            print("🔍 Searching for relevant information...")
            
            # Step 1: Search for relevant information
            search_results = self.search_agent.search_web(query)
            
            print("⚡ Generating code solution...")
            
            # Step 2: Generate code solution
            final_answer = self.code_agent.generate_code(query, context, search_results)
            
            print("✅ Code generation complete!")
            
            return final_answer
            
        except Exception as e:
            print(f"❌ Error in agent orchestrator: {str(e)}")
            return f"I encountered an error while processing your request: {str(e)}"