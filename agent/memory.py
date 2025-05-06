"""
Memory module for the webpage summarizer agent.
Handles conversation history and stores webpage summary.
"""

from langchain.memory import ConversationBufferWindowMemory
from typing import Dict, List, Any, Optional

class SummarizerMemory:
    """Memory component that stores conversation context and webpage summary."""
    
    def __init__(self, window_size: int = 3):
        """Initialize with specified window size for conversation history."""
        self.memory = ConversationBufferWindowMemory(
            k=window_size,
            memory_key="chat_history",
            return_messages=True,
            input_key="input"
        )
        self.current_summary = None
        self.current_url = None
        self.main_topic = None
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """Save the current conversation turn to memory."""
        self.memory.save_context(inputs, outputs)
    
    def load_memory_variables(self) -> Dict[str, Any]:
        """Load conversation history from memory."""
        return self.memory.load_memory_variables({})
    
    def set_summary(self, url: str, summary: str, topic: str) -> None:
        """Store the current webpage summary and its URL."""
        self.current_summary = summary
        self.current_url = url
        self.main_topic = topic
    
    def get_summary(self) -> Dict[str, Optional[str]]:
        """Retrieve the current summary information."""
        return {
            "url": self.current_url,
            "summary": self.current_summary,
            "main_topic": self.main_topic
        }
    
    def clear(self) -> None:
        """Clear all memory."""
        self.memory.clear()
        self.current_summary = None
        self.current_url = None
        self.main_topic = None

    def get_messages(self) -> List:
        """Get the raw message objects from memory for advanced processing."""
        chat_history = self.load_memory_variables().get("chat_history", [])
        return chat_history
    
    def get_formatted_history(self) -> str:
        """Get a nicely formatted string representation of the conversation history."""
        chat_history = self.get_messages()
        
        if not chat_history:
            return "No conversation history."
        
        formatted_history = ""
        for message in chat_history:
            if hasattr(message, 'type') and message.type == 'human':
                formatted_history += f"User: {message.content}\n\n"
            elif hasattr(message, 'type') and message.type == 'ai':
                formatted_history += f"Assistant: {message.content}\n\n"
        
        return formatted_history.strip()