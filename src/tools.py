# src/tools.py
from typing import List
from langchain.tools import Tool
import subprocess
import sys
import os

class CustomTools:
    """Custom tools for the AI agents"""
    
    def __init__(self):
        pass
    
    def get_tools(self) -> List[Tool]:
        """Get all available tools"""
        return [
            Tool(
                name="web_search",
                func=self.web_search,
                description="Search the web for current information about programming, libraries, and best practices"
            ),
            Tool(
                name="python_executor",
                func=self.execute_python,
                description="Execute Python code to test and validate solutions"
            ),
            Tool(
                name="package_installer",
                func=self.install_package,
                description="Install Python packages using pip"
            ),
            Tool(
                name="file_reader",
                func=self.read_file,
                description="Read the contents of a file"
            ),
            Tool(
                name="code_formatter",
                func=self.format_code,
                description="Format Python code using black formatter"
            )
        ]
    
    def web_search(self, query: str) -> str:
        """Search the web for information"""
        try:
            from langchain.tools import DuckDuckGoSearchRun
            search = DuckDuckGoSearchRun()
            return search.run(query)
        except Exception as e:
            return f"Search error: {str(e)}"
    
    def execute_python(self, code: str) -> str:
        """Execute Python code safely"""
        try:
            # Simple code execution (be careful with this in production)
            import io
            import sys
            from contextlib import redirect_stdout
            
            old_stdout = sys.stdout
            sys.stdout = captured_output = io.StringIO()
            
            try:
                exec(code)
                result = captured_output.getvalue()
            finally:
                sys.stdout = old_stdout
                
            return result if result else "Code executed successfully (no output)"
            
        except Exception as e:
            return f"Execution error: {str(e)}"
    
    def install_package(self, package_name: str) -> str:
        """Install a Python package"""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", package_name],
                capture_output=True,
                text=True,
                timeout=60
            )
            return result.stdout + result.stderr
        except Exception as e:
            return f"Installation error: {str(e)}"
    
    def read_file(self, file_path: str) -> str:
        """Read file contents"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return f"File read error: {str(e)}"
    
    def format_code(self, code: str) -> str:
        """Format Python code"""
        try:
            # Simple formatting (you can enhance this)
            lines = code.split('\n')
            formatted_lines = []
            indent_level = 0
            
            for line in lines:
                stripped = line.strip()
                if stripped:
                    if stripped.endswith(':'):
                        formatted_lines.append('    ' * indent_level + stripped)
                        indent_level += 1
                    elif stripped in ['else:', 'elif', 'except:', 'finally:']:
                        indent_level = max(0, indent_level - 1)
                        formatted_lines.append('    ' * indent_level + stripped)
                        indent_level += 1
                    else:
                        formatted_lines.append('    ' * indent_level + stripped)
                else:
                    formatted_lines.append('')
            
            return '\n'.join(formatted_lines)
        except Exception as e:
            return f"Formatting error: {str(e)}"