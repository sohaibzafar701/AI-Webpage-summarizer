"""
Prompt templates for the webpage summarizer agent.
These prompts are optimized for use with the Gemini API.
"""

from langchain.prompts import PromptTemplate

# Basic summarization prompt - optimized for Gemini
SUMMARIZATION_TEMPLATE = """
You are an expert content analyzer tasked with creating a concise, informative summary of a webpage.

INSTRUCTIONS:
1. Carefully analyze the webpage content provided below.
2. Focus on the most important information and key points.
3. Create a clear, well-structured summary that captures the essence of the webpage.
4. Ignore advertisements, navigation elements, and other irrelevant content.
5. Format your summary as a cohesive piece of text with logical flow.

WEBPAGE CONTENT:
{content}

SUMMARY:
"""

SUMMARIZATION_PROMPT = PromptTemplate(
    input_variables=["content"],
    template=SUMMARIZATION_TEMPLATE
)

# Main conversation agent prompt - optimized for Gemini
AGENT_TEMPLATE = """
You are a helpful AI assistant specializing in webpage summarization and information retrieval.

CONVERSATION HISTORY:
{chat_history}

CURRENT QUERY:
{input}

INSTRUCTIONS:
1. If the user is asking about a previously summarized webpage, provide information based on your knowledge of that webpage.
2. If asked about information not covered in the webpage, politely explain that you can only provide information contained in the summarized webpage.
3. Respond in a natural, conversational manner while being informative and accurate.
4. Keep your responses concise and focused on answering the specific question.

RESPONSE:
"""

AGENT_PROMPT = PromptTemplate(
    input_variables=["chat_history", "input"],
    template=AGENT_TEMPLATE
)

# Enhanced summarization prompt with structured approach - optimized for Gemini
ENHANCED_SUMMARIZATION_TEMPLATE = """
You are an expert content analyzer tasked with creating a high-quality summary of a webpage.

INSTRUCTIONS:
1. Analyze the webpage content below and identify:
   - The main topic or purpose
   - Key points and important details
   - Any significant conclusions or calls to action

2. Create a well-structured summary that:
   - Begins with a clear overview sentence
   - Includes the most important information in order of relevance
   - Maintains the original meaning without bias
   - Excludes advertisements, navigation elements, and irrelevant content

3. Format your response as a cohesive summary of 3-5 paragraphs.

WEBPAGE CONTENT:
{content}

SUMMARY:
"""

ENHANCED_SUMMARIZATION_PROMPT = PromptTemplate(
    input_variables=["content"],
    template=ENHANCED_SUMMARIZATION_TEMPLATE
)

# Topic extraction prompt - optimized for Gemini
TOPIC_EXTRACTION_TEMPLATE = """
Based on the following summary, identify the main topic of the webpage in 2-5 words.

SUMMARY:
{summary}

MAIN TOPIC (2-5 words only):
"""

TOPIC_EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["summary"],
    template=TOPIC_EXTRACTION_TEMPLATE
)

# Enhanced agent prompt with context - optimized for Gemini
ENHANCED_AGENT_TEMPLATE = """
You are a helpful AI assistant specializing in webpage analysis and information retrieval.

WEBPAGE SUMMARY:
{summary}

MAIN TOPIC: 
{main_topic}

CONVERSATION HISTORY:
{chat_history}

CURRENT QUERY:
{input}

INSTRUCTIONS:
1. Use the webpage summary and conversation history to provide an accurate, helpful response.
2. If asked about information not in the summary, politely explain you can only answer based on the summarized webpage.
3. If uncertain about details, acknowledge the limitations rather than making assumptions.
4. Keep your response focused, informative, and conversational.

RESPONSE:
"""

ENHANCED_AGENT_PROMPT = PromptTemplate(
    input_variables=["summary", "main_topic", "chat_history", "input"],
    template=ENHANCED_AGENT_TEMPLATE
)

# Relevance scoring prompt - optimized for Gemini's JSON capabilities
RELEVANCE_SCORING_TEMPLATE = """
Analyze the relevance and importance of the following webpage sections.

WEBPAGE SECTIONS:
{sections}

For each section, evaluate how relevant it is to the main content of the webpage, ignoring navigation elements, advertisements, and other non-content elements.

Rate each section on a scale of 0-10:
- 0: Completely irrelevant (navigation, ads, etc.)
- 5: Somewhat relevant but not essential
- 10: Highly relevant core content

INSTRUCTIONS:
1. For each section, provide a JSON object with:
   - score: The relevance score (0-10)
   - rationale: A brief explanation for the score
   - include_in_summary: Boolean (true if score >= 6, otherwise false)

2. Format your response as a valid JSON array of objects.

RESPONSE (valid JSON only):
"""

RELEVANCE_SCORING_PROMPT = PromptTemplate(
    input_variables=["sections"],
    template=RELEVANCE_SCORING_TEMPLATE
)