import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Initialize the FastAPI application
app = FastAPI(
    title="DevSecOps Assistant API", 
    description="A RAG-powered API for answering cybersecurity questions."
)

# 2. Define our Data Transfer Objects (DTOs) using Pydantic
class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str

# 3. Load the AI Brain globally so it doesn't reload on every single request
print("Initializing AI models and Vector DB...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = db.as_retriever(search_kwargs={"k": 3})

llm = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant"
)

system_prompt = (
    "You are an expert DevSecOps assistant. Use the provided context to answer the user's question. "
    "If you don't know the answer based on the context, just say that you don't know. "
    "Keep your answers highly technical, concise, and focused on cybersecurity mitigation strategies.\n\n"
    "Context:\n{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt), 
    ("human", "{input}")
])

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 4. Create the POST Endpoint
@app.post("/ask", response_model=AskResponse)
def ask_assistant(request: AskRequest):
    try:
        # Pass the incoming question to our RAG chain
        response = rag_chain.invoke({"input": request.question})
        return {"answer": response["answer"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")