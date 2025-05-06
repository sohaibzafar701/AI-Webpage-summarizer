# AI Webpage Summarizer 

A LangChain-based webpage summarization agent powered by Google's Gemini API.

## Features

- **Webpage Summarization**: Extracts and summarizes content from any URL
- **Gemini API Integration**: Uses Google's state-of-the-art language models
- **Conversation Memory**: Remembers the previous 3 interactions for contextual follow-up questions
- **FastAPI Endpoint**: Provides a clean API interface for integrating with other applications
- **Enhanced Summarization**: Uses optimized prompts for high-quality, relevant summaries

## Project Structure

```
AI-Webpage-Summarizer/
├── .env                  # Environment variables
├── requirements.txt      # Dependencies
├── app.py                # FastAPI application
├── agent/
│   ├── __init__.py
│   ├── browser.py        # Web browser tool
│   ├── memory.py         # Conversation memory
│   ├── prompts.py        # Prompt templates
│   └── summarizer.py     # Main summarization agent
└── README.md             # Project documentation
```

## Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/sohaibzafar701/AI-Webpage-summarizer.git
cd AI-Webpage-summarizer
```

2. **Set up a virtual environment**

```bash
python -m venv venv
venv\Scripts\activate  # On linux, use: source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Set up Gemini API access**

- Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
- Create a Gemini API key
- Save it to your .env file

5. **Configure environment variables**

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_gemini_api_key_here
MODEL_NAME=gemini-1.5-pro
```

6. **Run the application**

```bash
uvicorn app:app --reload
```

The API will be available at `http://localhost:8000`.

## API Usage

### Summarize a Webpage

**Endpoint:** `POST /summarize`

**Request:**
```json
{
  "url": "https://example.com"
}
```

**Response:**
```json
{
  "summary": "The webpage discusses the role of AI in modern education systems...",
  "main_topic": "AI in Education",
  "url": "https://example.com"
}
```

### Ask a Follow-up Question

**Endpoint:** `POST /ask`

**Request:**
```json
{
  "question": "What are the key benefits mentioned?"
}
```

**Response:**
```json
{
  "answer": "The key benefits mentioned include personalized learning experiences, reduced administrative workload for teachers, and improved learning outcomes through adaptive content delivery."
}
```

### Get Current Summary

**Endpoint:** `GET /current`

**Response:**
```json
{
  "summary": "The webpage discusses the role of AI in modern education systems...",
  "main_topic": "AI in Education",
  "url": "https://example.com"
}
```

## Gemini-Specific Considerations

### Model Selection

The solution uses `gemini-1.5-pro` by default, which offers a good balance of quality and speed. Other options include:

- `gemini-1.5-flash`: Faster responses, smaller context window
- `gemini-1.5-pro-preview`: The latest model with maximum capabilities

