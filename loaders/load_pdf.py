import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone as LangchainPinecone
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain.schema import Document
from pinecone import Pinecone, ServerlessSpec
import re

load_dotenv()


def split_legal_sentences(text):
    # Breaks on Articles, section headers, bullets, and paragraphs
    pattern = r'(Article \d+[\s\S]*?)(?=Article \d+|$)|(?<=\n)\d{1,2}\. .*?(?=\n\d{1,2}\. |\Z)|(?<=\n)[a-z]\) .*?(?=\n[a-z]\) |\Z)'
    
    # Match all structured parts
    matches = re.findall(pattern, text)
    
    # Fallback: split large pieces into logical chunks
    refined = []
    for match in matches:
        refined.extend([s.strip() for s in re.split(r'\n{2,}|\.\s+', match) if s.strip()])
    
    return refined


def semantic_chunking(documents, max_chunk_size=1000, similarity_threshold=0.75):
    """ 
    Chunk documents based on semantic similarity instead of token size alone.
    Uses sentence embeddings and cosine similarity to keep context coherent.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight, fast embedding model
    chunks = []

    for doc in documents:
        text = doc.page_content
        sentences = split_legal_sentences(text)  #Split based on regex 
        embeddings = model.encode(sentences)  # Create vector for each sentence

        current_chunk = [sentences[0]]
        current_length = len(sentences[0])

        for i in range(1, len(sentences)):
            # Measure similarity with previous sentence
            sim = cosine_similarity([embeddings[i]], [embeddings[i - 1]])[0][0]

            # Split chunk if similarity is low (topic likely shifted) or it's too long
            if sim < similarity_threshold or current_length > max_chunk_size:
                chunks.append(". ".join(current_chunk).strip())
                current_chunk = []
                current_length = 0

            current_chunk.append(sentences[i])
            current_length += len(sentences[i])

        # Catch last leftover chunk
        if current_chunk:
            chunks.append(". ".join(current_chunk).strip())

    # Return as LangChain Document objects
    return [Document(page_content=chunk) for chunk in chunks]

def load_eu_act_and_store(pdf_path: str, index_name: str = "eu-ai-act-index"):
    # Load EU AI Act PDF as LangChain Documents
    loader = PyPDFLoader(pdf_path)
    raw_docs = loader.load()

    # Use semantic chunking
    chunks = semantic_chunking(raw_docs)

    # Set up embedding model
    embeddings = OpenAIEmbeddings()

    # Initialize Pinecone (v3 client)
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    # Create index if not exists
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=os.getenv("PINECONE_ENV"))
        )

    # Get index and store documents
    index = pc.Index(index_name)

    # Store in Pinecone via LangChain
    LangchainPinecone.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=index_name
    )

    print(f"Embedded and stored {len(chunks)} semantically chunked sections from EU AI Act in Pinecone.")
