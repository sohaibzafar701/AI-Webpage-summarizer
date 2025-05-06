"""
WebBrowserTool for fetching and extracting web content.
This module handles webpage access and content extraction.
"""

from typing import Dict, List, Optional, Any
from langchain.tools import BaseTool
import httpx
from bs4 import BeautifulSoup

class WebBrowserTool(BaseTool):
    """Tool for browsing websites and extracting their content."""
    
    # Add type annotations to these class attributes to fix the Pydantic error
    name: str = "web_browser"
    description: str = "Useful for fetching and extracting content from a webpage given its URL."
    
    def _run(self, url: str) -> str:
        """Use the tool with a URL."""
        try:
            # Basic validation
            if not url.startswith(('http://', 'https://')):
                return "Error: URL must start with http:// or https://"
            
            # Fetch the webpage
            response = httpx.get(url, follow_redirects=True, timeout=10.0)
            response.raise_for_status()
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract main content (remove scripts, styles, etc.)
            for script in soup(["script", "style", "meta", "noscript", "iframe"]):
                script.extract()
                
            # Get text
            text = soup.get_text(separator=' ', strip=True)
            
            # Clean up excessive whitespace
            import re
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Truncate if too long
            if len(text) > 50000:
                text = text[:50000] + "...[content truncated due to length]"
                
            return text
        except Exception as e:
            return f"Error accessing URL: {str(e)}"
    
    async def _arun(self, url: str) -> str:
        """Async version of run."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, follow_redirects=True, timeout=10.0)
                response.raise_for_status()
                
                # Process with BeautifulSoup (same as _run)
                soup = BeautifulSoup(response.text, 'html.parser')
                
                for script in soup(["script", "style", "meta", "noscript", "iframe"]):
                    script.extract()
                    
                text = soup.get_text(separator=' ', strip=True)
                import re
                text = re.sub(r'\s+', ' ', text).strip()
                
                if len(text) > 50000:
                    text = text[:50000] + "...[content truncated due to length]"
                    
                return text
        except Exception as e:
            return f"Error accessing URL: {str(e)}"