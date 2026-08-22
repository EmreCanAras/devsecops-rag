from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

print("1. PDF loading...")
loader = PyPDFLoader("data/OWASP Top 10 2025 by Miglen Evlogiev.pdf")
documents = loader.load()

print("2. Text parsing(Chunking)...")
# Her biri ortalama 1000 karakterlik parçalar oluşturuyoruz. 
# Cümleler ortadan kesilmesin diye 100 karakterlik bir örtüşme (overlap) bırakıyoruz.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)

print("3. AI model and vectors have generating(Embedding)...")
# M1 işlemcilerde gayet hızlı ve verimli çalışan hafif bir yerel model kullanıyoruz.
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("4. Saving to ChromaDB...")
# Veritabanını proje klasörü içindeki 'chroma_db' dizininde oluşturur.
db = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")

print(f"Fin! Total {len(chunks)} chunks has stored to the database")