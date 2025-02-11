from fastapi import FastAPI, UploadFile, File, HTTPException
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from together import Together
import os
import shutil

API_KEY = ""

app = FastAPI()
client = Together(api_key=API_KEY)

# Embeddings Model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Directory for Uploaded Files
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# FAISS Vector Store
vector_store = None


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    """process pdf or txt file and convert to embeddings -> store in FAISS"""
    global vector_store
    file_path = os.path.join(UPLOAD_DIR, file.filename)

    # Save uploaded file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Load file content
    if file.filename.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file.filename.endswith(".txt"):
        loader = TextLoader(file_path)
    else:
        raise HTTPException(status_code=400, detail="File type not supported. Support .pdf and .txt")

    docs = loader.load()

    # Split text into chunks for embedding
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(docs)

    # Store embeddings in FAISS
    vector_store = FAISS.from_documents(chunks, embeddings)

    return {"message": "File uploaded and processed successfully."}


@app.post("/ask/")
async def ask_question(query: str):
    global vector_store

    if vector_store is None:
        raise HTTPException(status_code=400, detail="No file uploaded yet.")

    # Retrieve relevant text chunks
    retriever = vector_store.as_retriever()
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    # use either Together AI Llama-3.3 70B Model or Mistral-7B-Instruct-v0.1
    response = client.chat.completions.create(
        model="mistralai/Mistral-7B-Instruct-v0.1",
        messages=[
            {"role": "system", "content": "You are an expert assistant answering user questions based on provided documents."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ],
    )

    return {"response": response.choices[0].message.content}
