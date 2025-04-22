import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pymongo import MongoClient
import re

load_dotenv()

def extract_metadata_from_chunk(text_chunk: str) -> dict:
    """Extract metadata from a chunk using regex-based parsing."""
    doc = {}

    def find_value(label, default="N/A"):
        pattern = rf"{label}:\s*(.*)"
        match = re.search(pattern, text_chunk, re.IGNORECASE)
        return match.group(1).strip() if match else default

    doc['system_name'] = find_value("System Name", "Unknown")
    doc['provider'] = find_value("Provider", "Unknown")
    doc['use_case_description'] = find_value("Use Case Description")
    doc['target_users'] = find_value("Target Users")
    doc['environment'] = find_value("Deployment Environment")
    doc['core_functionality'] = find_value("Core Functionality")
    doc['compliance_notes'] = find_value("Compliance Notes")
    doc['biometric_data_used'] = 'Yes' if "biometric" in text_chunk.lower() else 'No'
    doc['is_gpaai'] = 'Yes' if "general purpose ai" in text_chunk.lower() else 'No'

    return doc


def store_alas_to_mongo(pdf_path: str):
    """Load, chunk, extract metadata, and store into MongoDB."""
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    # Smart chunking (based on logical paragraph structure)
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
    chunks = splitter.split_documents(pages)

    # We only need 1 chunk likely (table usually on one page)
    combined_text = " ".join([chunk.page_content for chunk in chunks])

    metadata = extract_metadata_from_chunk(combined_text)

    # MongoDB connection
    client = MongoClient(os.getenv("MONGO_URI"))
    db = client["eu_ai_compliance"]["alas_documents"]
    db.insert_one(metadata)

    print(" ALAS metadata extracted and stored in MongoDB successfully.")

