import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Connect to our existing local database
print("Connecting to ChromaDB...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# 2. Set up the retriever (fetch top 3 most relevant chunks)
retriever = db.as_retriever(search_kwargs={"k": 3})

# 3. Initialize the LLM (using Llama 3 on Groq for extreme speed)
llm = ChatGroq(
    api_key=os.environ.get("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant" 
)

# 4. Define the System Prompt
system_prompt = (
    "You are an expert DevSecOps assistant. Use the provided context to answer the user's question. "
    "If you don't know the answer based on the context, just say that you don't know. "
    "Keep your answers highly technical, concise, and focused on cybersecurity mitigation strategies.\n\n"
    "Context:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# 5. Build the RAG Chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 6. Test the Assistant
user_question = input("Ask your question to the DevSecOps assistant: ")
print("Thinking...")

response = rag_chain.invoke({"input": user_question})

print("\n--- ASSISTANT RESPONSE ---")
print(response["answer"])