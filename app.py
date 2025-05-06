"""
FastAPI application for the AICO Webpage Summarizer.
Provides API endpoints for summarizing webpages and asking follow-up questions.
"""

import os
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, Field
from typing import Dict, Optional, List
from dotenv import load_dotenv

from agent.summarizer import WebpageSummarizer

# Load environment variables
load_dotenv()

# Get API key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set")

# Get optional model name or use default
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-1.5-pro")

# Initialize summarizer agent
summarizer = WebpageSummarizer(api_key=GOOGLE_API_KEY, model=MODEL_NAME)

# Create FastAPI app
app = FastAPI(
    title="AICO Webpage Summarizer",
    description="An API for summarizing webpages and extracting their main topics using Google's Gemini API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request models
class SummarizeRequest(BaseModel):
    url: HttpUrl
    
    class Config:
        json_schema_extra = {
            "example": {
                "url": "https://example.com"
            }
        }

class QuestionRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    
    class Config:
        json_schema_extra = {
            "example": {
                "question": "What is the main point of the webpage?"
            }
        }

# Define response models
class SummaryResponse(BaseModel):
    summary: str
    main_topic: str
    url: HttpUrl

class ErrorResponse(BaseModel):
    error: str

class QuestionResponse(BaseModel):
    answer: str

class StatusResponse(BaseModel):
    status: str
    message: Optional[str] = None

# Define API endpoints
@app.post("/summarize", response_model=SummaryResponse, responses={400: {"model": ErrorResponse}})
async def summarize_webpage(request: SummarizeRequest):
    """Summarize a webpage and extract its main topic."""
    try:
        result = summarizer.summarize_url(str(request.url))
        
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
            
        return {
            "summary": result["summary"],
            "main_topic": result["main_topic"],
            "url": request.url
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/ask", response_model=QuestionResponse, responses={400: {"model": ErrorResponse}})
async def ask_question(request: QuestionRequest):
    """Ask a question about the previously summarized webpage."""
    summary_info = summarizer.get_current_summary()
    
    if not summary_info["summary"]:
        raise HTTPException(
            status_code=400, 
            detail="No webpage has been summarized yet. Please summarize a webpage first."
        )
        
    try:
        answer = summarizer.answer_question(request.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/current", response_model=Optional[SummaryResponse])
async def get_current_summary():
    """Get the currently stored webpage summary."""
    summary_info = summarizer.get_current_summary()
    
    if not summary_info["summary"]:
        return None
        
    return {
        "summary": summary_info["summary"],
        "main_topic": summary_info["main_topic"],
        "url": summary_info["url"]
    }

@app.post("/clear", response_model=StatusResponse)
async def clear_memory():
    """Clear the conversation memory and current summary."""
    try:
        summarizer.clear_memory()
        return {"status": "success", "message": "Memory cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health", response_model=StatusResponse)
async def health_check():
    """Check if the API is operational."""
    return {"status": "healthy"}

# Run the app
if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    
    uvicorn.run(app, host=host, port=port)