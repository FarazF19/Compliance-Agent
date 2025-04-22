import os
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import pinecone

load_dotenv()

def semantic_chunking(documents):
    """ Use recursive chunking with smart overlap for legal context. """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=300,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    return splitter.split_documents(documents)

def load_eu_act_and_store(pdf_path: str, index_name: str = "eu-ai-act-index"):
    # Load EU AI Act PDF
    loader = PyPDFLoader(pdf_path)
    raw_docs = loader.load()

    # Apply smarter chunking
    chunks = semantic_chunking(raw_docs)

    # Setup Pinecone + Embeddings
    pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))
    embeddings = OpenAIEmbeddings()

    if index_name not in pinecone.list_indexes():
        pinecone.create_index(name=index_name, dimension=1536)

    # Store chunks in Pinecone
    Pinecone.from_documents(chunks, embeddings, index_name=index_name)
    print(f"✅ Embedded and stored {len(chunks)} chunks from EU AI Act in Pinecone.")
