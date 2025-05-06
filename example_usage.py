"""
Example usage of the AICO Webpage Summarizer with Gemini.

This script demonstrates how to use the Webpage Summarizer agent
both interactively and through the API.
"""

import os
from dotenv import load_dotenv
import uvicorn
import asyncio
import httpx
import json

# Load environment variables
load_dotenv()

def interactive_demo():
    """Run the summarizer in interactive mode."""
    from agent.summarizer import WebpageSummarizer
    
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("Error: GOOGLE_API_KEY environment variable is not set")
        return
    
    model = os.getenv("MODEL_NAME", "gemini-1.5-pro")
    
    print(f"=== AICO Webpage Summarizer (Gemini {model}) ===")
    print("Enter a URL to summarize or 'q' to quit")
    
    # Initialize the summarizer
    summarizer = WebpageSummarizer(api_key=api_key, model=model)
    
    while True:
        user_input = input("\nEnter URL or question (or 'q' to quit): ")
        
        if user_input.lower() == 'q':
            break
        
        # Check if input is a URL or a question
        if user_input.startswith(("http://", "https://")):
            print("\nFetching and summarizing webpage...")
            result = summarizer.summarize_url(user_input)
            
            if "error" in result:
                print(f"Error: {result['error']}")
            else:
                print(f"\nMain Topic: {result['main_topic']}")
                print(f"\nSummary:\n{result['summary']}")
        else:
            # Treat as a question
            print("\nAnswering question...")
            if not summarizer.get_current_summary()["summary"]:
                print("No webpage has been summarized yet. Please enter a URL first.")
            else:
                answer = summarizer.answer_question(user_input)
                print(f"\nAnswer: {answer}")

async def api_demo():
    """Demo the agent through the API."""
    # Ensure the API is running first (run uvicorn app:app in a separate terminal)
    
    base_url = "http://localhost:8000"
    
    async with httpx.AsyncClient() as client:
        # Test URL summarization
        test_url = "https:Janjuaweb.com"
        
        print("\n=== AICO Webpage Summarizer API Demo ===")
        print(f"Summarizing URL: {test_url}")
        
        try:
            response = await client.post(
                f"{base_url}/summarize",
                json={"url": test_url},
                timeout=60.0  # Longer timeout for summarization
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"\nMain Topic: {result['main_topic']}")
                print(f"\nSummary:\n{result['summary']}")
                
                # Test asking a question
                test_question = "What are the key applications of AI mentioned?"
                print(f"\nAsking: {test_question}")
                
                question_response = await client.post(
                    f"{base_url}/ask",
                    json={"question": test_question}
                )
                
                if question_response.status_code == 200:
                    answer = question_response.json()
                    print(f"\nAnswer: {answer['answer']}")
                else:
                    print(f"Error: {question_response.status_code} - {question_response.text}")
            else:
                print(f"Error: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"Error: {str(e)}")
            print("Make sure the API server is running (uvicorn app:app)")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--api":
        asyncio.run(api_demo())
    else:
        interactive_demo()