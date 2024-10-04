from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
from typing import List, Dict
import uuid

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set environment variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["PINECONE_API_KEY"] = os.getenv("PINECONE_API_KEY")

# Initialize embeddings and vector store
index_name = "kalki"
embeddings = OpenAIEmbeddings(api_key=os.environ["OPENAI_API_KEY"])

vectorstore = None
try:
    vectorstore = PineconeVectorStore.from_existing_index(index_name, embeddings)
except Exception as e:
    print(f"Error creating vectorstore: {e}")

# Initialize QA chain
qa_chain = None
if vectorstore:
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(api_key=os.environ["OPENAI_API_KEY"]),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=False
    )

# Store chat histories
chat_histories: Dict[str, List[Dict[str, str]]] = {}

# Define request and response models
class QueryRequest(BaseModel):
    query: str
    chat_id: str = None

class QueryResponse(BaseModel):
    result: str
    chat_id: str

class ChatHistoryResponse(BaseModel):
    chat_histories: Dict[str, List[Dict[str, str]]]

# Endpoint to process queries
@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest) -> QueryResponse:
    if vectorstore is None:
        raise HTTPException(status_code=500, detail="Vector store not initialized")

    try:
        llm_response = qa_chain(request.query)
        result = llm_response['result']

        # Create a new chat if chat_id is not provided
        if not request.chat_id:
            chat_id = str(uuid.uuid4())
            chat_histories[chat_id] = []
        else:
            chat_id = request.chat_id

        # Add the query and result to the chat history
        chat_histories[chat_id].append({"query": request.query, "result": result})

        return QueryResponse(result=result, chat_id=chat_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint to get chat histories
@app.get("/chat_histories", response_model=ChatHistoryResponse)
async def get_chat_histories() -> ChatHistoryResponse:
    return ChatHistoryResponse(chat_histories=chat_histories)

# Endpoint to start a new chat
@app.post("/new_chat")
async def new_chat():
    chat_id = str(uuid.uuid4())
    chat_histories[chat_id] = []
    return {"chat_id": chat_id}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)