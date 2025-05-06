"""
Main summarizer agent implementation using Google's Gemini API.
"""

from typing import Dict, List, Tuple, Any, Optional
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from .browser import WebBrowserTool
from .memory import SummarizerMemory
from .prompts import (
    SUMMARIZATION_PROMPT, 
    AGENT_PROMPT,
    TOPIC_EXTRACTION_PROMPT
)

class WebpageSummarizer:
    """Agent that summarizes webpages and answers questions about them using Gemini."""
    
    def __init__(self, api_key: str, model: str = "gemini-1.5-pro"):
        """Initialize the summarizer with Google API key and model."""
        # Initialize the Gemini LLM
        self.llm = ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model=model,
            temperature=0,
            top_p=0.95,
            top_k=40
        )
        
        # Initialize tools and memory
        self.browser_tool = WebBrowserTool()
        self.memory = SummarizerMemory(window_size=3)
        
        # Create summarization chain
        self.summarization_chain = LLMChain(
            llm=self.llm,
            prompt=SUMMARIZATION_PROMPT,
            output_parser=StrOutputParser()
        )
        
        # Create topic extraction chain
        self.topic_extraction_chain = LLMChain(
            llm=self.llm,
            prompt=TOPIC_EXTRACTION_PROMPT,
            output_parser=StrOutputParser()
        )
        
        # Create conversation chain
        self.conversation_chain = LLMChain(
            llm=self.llm,
            prompt=AGENT_PROMPT
        )
        
        # Create tools list
        self.tools = [self.browser_tool]
    
    def _extract_main_topic(self, summary: str) -> str:
        """Extract the main topic from a summary."""
        try:
            # Use the dedicated topic extraction chain
            topic = self.topic_extraction_chain.run(summary=summary)
            return topic.strip()
        except Exception as e:
            # Fallback with direct prompting if chain fails
            topic_prompt = f"Based on this summary, what is the single main topic in 2-5 words?\n\n{summary}"
            topic_messages = [HumanMessage(content=topic_prompt)]
            response = self.llm.invoke(topic_messages)
            return response.content.strip()
    
    def summarize_url(self, url: str) -> Dict[str, str]:
        """Summarize a webpage given its URL."""
        try:
            # Fetch webpage content
            content = self.browser_tool._run(url)
            
            if content.startswith("Error"):
                return {"error": content}
            
            # Handle large content for Gemini's context window
            if len(content) > 30000:
                content = content[:30000] + "...[content truncated due to length]"
            
            # Generate summary
            summary = self.summarization_chain.run(content=content)
            
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
        """Answer a question about the summarized webpage."""
        try:
            # Get chat history from memory
            memory_vars = self.memory.load_memory_variables()
            
            # Get response from conversation chain
            response = self.conversation_chain.run(
                chat_history=memory_vars.get("chat_history", ""),
                input=question
            )
            
            # Save context to memory
            self.memory.save_context(
                {"input": question},
                {"output": response}
            )
            
            return response
        except Exception as e:
            # Graceful error handling for Gemini API errors
            error_msg = str(e)
            if "content_blocked" in error_msg.lower():
                return "I'm unable to answer this question due to content restrictions. Please try rephrasing your question."
            elif "quota_exceeded" in error_msg.lower():
                return "The API usage limit has been reached. Please try again later."
            else:
                return f"An error occurred while processing your question: {error_msg}"
    
    def get_current_summary(self) -> Dict[str, Optional[str]]:
        """Get the current webpage summary."""
        return self.memory.get_summary()
    
    def clear_memory(self) -> None:
        """Clear the conversation memory and current summary."""
        self.memory.clear()
        return {"status": "Memory cleared successfully"}