"""
AICO Webpage Summarizer Agent
----------------------------

A LangChain-based agent that summarizes webpages using Google's Gemini API.

This package provides tools for webpage content extraction, summarization,
conversation memory, and a FastAPI interface.
"""

from .browser import WebBrowserTool
from .memory import SummarizerMemory
from .summarizer import WebpageSummarizer
from .prompts import (
    SUMMARIZATION_PROMPT,
    AGENT_PROMPT,
    ENHANCED_SUMMARIZATION_PROMPT,
    ENHANCED_AGENT_PROMPT,
    TOPIC_EXTRACTION_PROMPT,
    RELEVANCE_SCORING_PROMPT
)

__all__ = [
    'WebBrowserTool',
    'SummarizerMemory',
    'WebpageSummarizer',
    'SUMMARIZATION_PROMPT',
    'AGENT_PROMPT',
    'ENHANCED_SUMMARIZATION_PROMPT',
    'ENHANCED_AGENT_PROMPT',
    'TOPIC_EXTRACTION_PROMPT',
    'RELEVANCE_SCORING_PROMPT'
]

__version__ = "0.1.0"