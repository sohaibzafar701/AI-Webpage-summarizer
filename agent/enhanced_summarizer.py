# agent/enhanced_summarizer.py
from typing import Dict, List, Tuple, Any, Optional
import json
from bs4 import BeautifulSoup
import re
from langchain.chains import LLMChain
from langchain_google_genai import ChatGoogleGenerativeAI  # Import Gemini chat model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain.agents import initialize_agent, AgentType
from langchain_core.runnables import RunnableSerializable

from .browser import WebBrowserTool
from .memory import SummarizerMemory
from .improved_prompts import (
    ENHANCED_SUMMARIZATION_PROMPT,
    ENHANCED_AGENT_PROMPT,
    TOPIC_EXTRACTION_PROMPT,
    RELEVANCE_SCORING_PROMPT
)

class EnhancedWebpageSummarizer:
    """Enhanced agent that produces higher quality webpage summaries with content prioritization."""
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-pro"):
        """Initialize the summarizer with Google API key and optional model."""
        # We use Gemini's larger model for better handling of large webpages
        self.llm = ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model=model,
            temperature=0,
            # Gemini-specific parameters
            top_p=0.95,
            top_k=40
        )
        self.browser_tool = WebBrowserTool()
        self.memory = SummarizerMemory(window_size=3)
        
        # Create summarization chain with enhanced prompt
        self.summarization_chain = LLMChain(
            llm=self.llm,
            prompt=ENHANCED_SUMMARIZATION_PROMPT,
            output_parser=StrOutputParser()
        )
        
        # Create topic extraction chain
        self.topic_extraction_chain = LLMChain(
            llm=self.llm,
            prompt=TOPIC_EXTRACTION_PROMPT,
            output_parser=StrOutputParser()
        )
        
        # Create conversation chain with enhanced prompt
        self.conversation_chain = LLMChain(
            llm=self.llm,
            prompt=ENHANCED_AGENT_PROMPT
        )
        
        # Create tools list
        self.tools = [self.browser_tool]
    
    def _extract_sections(self, html_content: str) -> List[str]:
        """Extract meaningful sections from HTML content."""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove scripts, styles, etc.
        for element in soup(['script', 'style', 'meta', 'noscript', 'iframe']):
            element.extract()
        
        sections = []
        
        # Try to find content sections - prioritize semantic HTML5 elements
        content_elements = soup.find_all(['article', 'section', 'main', 'div.content', 'div.main'])
        
        if content_elements:
            # Use identified content sections
            for element in content_elements:
                if len(element.get_text(strip=True)) > 100:  # Ignore very short sections
                    sections.append(element.get_text(separator=' ', strip=True))
        else:
            # Fallback to paragraph-based extraction
            paragraphs = soup.find_all('p')
            
            # Group paragraphs into logical sections
            current_section = []
            section_text = ""
            
            for p in paragraphs:
                text = p.get_text(strip=True)
                if text and len(text) > 20:  # Skip very short paragraphs (likely not content)
                    current_section.append(text)
                    section_text += " " + text
                    
                    # If section is getting long or there's a natural break, create a new section
                    if len(section_text) > 2000 or text.endswith('.') and p.find_next_sibling('h2'):
                        sections.append(" ".join(current_section))
                        current_section = []
                        section_text = ""
            
            # Add the last section if not empty
            if current_section:
                sections.append(" ".join(current_section))
        
        # If we still don't have meaningful sections, use a fallback approach
        if not sections:
            # Get all text and split by headers or significant breaks
            text = soup.get_text(separator=' ', strip=True)
            # Clean up excessive whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            
            # Split into manageable chunks if too large
            if len(text) > 5000:
                chunks = []
                start = 0
                while start < len(text):
                    chunks.append(text[start:start+3000])
                    start += 3000
                sections = chunks
            else:
                sections = [text]
        
        return sections
    
    def _score_content_relevance(self, sections: List[str]) -> List[Dict]:
        """Score webpage sections by relevance for better content prioritization."""
        if not sections:
            return []
            
        # For efficiency, limit to a reasonable number of sections
        sections_to_analyze = sections[:10] if len(sections) > 10 else sections
        
        # Format sections for the prompt
        sections_text = "\n".join([f"SECTION {i+1}:\n{section[:500]}...\n" 
                                 for i, section in enumerate(sections_to_analyze)])
        
        # Get relevance scores
        try:
            messages = [
                SystemMessage(content="You are a content relevance analyst."),
                HumanMessage(content=RELEVANCE_SCORING_PROMPT.format(sections=sections_text))
            ]
            response = self.llm.invoke(messages)
            
            # Extract JSON from the response
            json_str = response.content.strip()
            # Handle case where model wraps JSON in ```json ... ``` format
            if json_str.startswith('```') and json_str.endswith('```'):
                json_str = json_str.split('```')[1]
                if json_str.startswith('json'):
                    json_str = json_str[4:].strip()
            
            return json.loads(json_str)
        except Exception as e:
            # Fallback in case of parsing errors
            return [{"score": 8, "rationale": "Automatic fallback scoring", 
                     "include_in_summary": True} for _ in sections_to_analyze]
    
    def _extract_main_topic(self, summary: str) -> str:
        """Extract the main topic from a summary using the topic extraction chain."""
        topic = self.topic_extraction_chain.run(summary=summary)
        return topic.strip()
    
    def summarize_url(self, url: str) -> Dict[str, str]:
        """Summarize a webpage given its URL using the enhanced approach."""
        try:
            # Fetch webpage content
            raw_content = self.browser_tool._run(url)
            
            if raw_content.startswith("Error"):
                return {"error": raw_content}
            
            # Extract and score sections - handle Gemini's context window limits
            sections = self._extract_sections(raw_content)
            scored_sections = self._score_content_relevance(sections)
            
            # Filter to relevant sections
            relevant_sections = []
            for i, score_data in enumerate(scored_sections):
                if i < len(sections) and score_data.get("include_in_summary", False):
                    relevant_sections.append(sections[i])
            
            # If no sections were deemed relevant, use all sections
            if not relevant_sections and sections:
                relevant_sections = sections
            
            # Combine relevant sections for summarization
            content_to_summarize = "\n\n".join(relevant_sections)
            
            # Adjust based on Gemini's context limits - may need to be reduced for some Gemini models
            if len(content_to_summarize) > 12000:  
                content_to_summarize = content_to_summarize[:12000]
            
            # Generate summary
            summary = self.summarization_chain.run(content=content_to_summarize)
            
            # Extract main topic
            main_topic = self._extract_main_topic(summary)
            
            # Store in memory
            self.memory.set_summary(url, summary, main_topic)
            
            return {
                "url": url,
                "summary": summary,
                "main_topic": main_topic
            }
        except Exception as e:
            return {"error": f"Error summarizing webpage: {str(e)}"}
    
    def answer_question(self, question: str) -> str:
        """Answer a question about the summarized webpage with improved context awareness."""
        # Get chat history from memory
        memory_vars = self.memory.load_memory_variables()
        
        # Get current summary info
        summary_info = self.memory.get_summary()
        
        # Get response from conversation chain with enhanced context
        response = self.conversation_chain.run(
            chat_history=memory_vars.get("chat_history", ""),
            summary=summary_info.get("summary", "No webpage has been summarized yet."),
            main_topic=summary_info.get("main_topic", "Unknown"),
            input=question
        )
        
        # Save context to memory
        self.memory.save_context(
            {"input": question},
            {"output": response}
        )
        
        return response
    
    def get_current_summary(self) -> Dict[str, str]:
        """Get the current webpage summary."""
        return self.memory.get_summary()