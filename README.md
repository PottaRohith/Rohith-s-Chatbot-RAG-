
# FastAPI Chatbot with LangChain, OpenAI, and Pinecone

This project implements a FastAPI server that processes natural language queries using OpenAI embeddings and Pinecone for vector storage and retrieval. The chatbot allows users to start new chats, store chat histories, and retrieve responses using the LangChain framework.

## Features
- **Query Processing**: Handles user queries and returns responses using OpenAI's language model.
- **Pinecone Vector Store**: Stores and retrieves vectorized queries and responses.
- **Chat Histories**: Stores chat histories by chat ID, allowing users to resume chats.
- **API Endpoints**: Provides multiple endpoints to start new chats, submit queries, and retrieve chat histories.

## Prerequisites
Before running this project, ensure you have the following:
- Python 3.8 or above
- OpenAI API key
- Pinecone API key
- Pinecone index created (named `kalki` in this example)
- A `.env` file containing your API keys:
  ```
  OPENAI_API_KEY=your_openai_api_key
  PINECONE_API_KEY=your_pinecone_api_key
  ```

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/PottaRohith/Rohith-s-Chatbot-RAG-
   ```

2. Install the required Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the root directory and add your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   PINECONE_API_KEY=your_pinecone_api_key
   ```

4. Ensure your Pinecone index (`kalki`) is already set up and accessible.

## Running the Application

To start the FastAPI server, run the following command:
```bash
uvicorn main:app --reload
```

The API will be accessible at `http://127.0.0.1:8000`.

## API Endpoints

### 1. POST `/query`
Process a user query and return a response.
- **Request Body**:
  ```json
  {
    "query": "Your question here",
    "chat_id": "Optional, existing chat ID"
  }
  ```
- **Response**:
  ```json
  {
    "result": "Response from OpenAI",
    "chat_id": "Chat ID"
  }
  ```

### 2. GET `/chat_histories`
Retrieve the chat histories for all chats.
- **Response**:
  ```json
  {
    "chat_histories": {
      "chat_id1": [
        {"query": "User query", "result": "Bot response"}
      ]
    }
  }
  ```

### 3. POST `/new_chat`
Start a new chat session and generate a new `chat_id`.
- **Response**:
  ```json
  {
    "chat_id": "Generated chat ID"
  }
  ```

## Project Structure

```
.
├── main.py            # Main FastAPI application
├── requirements.txt   # Python dependencies
├── .env               # Environment variables (API keys)
└── README.md          # Project documentation
```

## Dependencies

- **FastAPI**: Web framework for building the API.
- **LangChain**: Provides LLM and retrieval logic.
- **Pinecone**: Vector database to store and retrieve embeddings.
- **OpenAI**: API for language model responses.
- **Uvicorn**: ASGI server to run the FastAPI app.

## Environment Variables

Make sure to set the following environment variables in your `.env` file:
- `OPENAI_API_KEY`: Your OpenAI API key.
- `PINECONE_API_KEY`: Your Pinecone API key.


