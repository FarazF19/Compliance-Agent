from pymongo import MongoClient
import fitz  # PyMuPDF
import os

def extract_alas_metadata(text: str):
    lines = text.split('\n')
    doc = {}
    for line in lines:
        if 'System Name' in line:
            doc['system_name'] = line.split(':')[-1].strip()
        elif 'Use Case' in line:
            doc['use_case'] = line.split(':')[-1].strip()
        elif 'Sensitive Data' in line:
            doc['sensitive_data'] = 'yes' in line.lower()
    return doc

def parse_pdf(path):
    doc = fitz.open(path)
    return "\n".join([page.get_text() for page in doc])

def store_alas_to_mongo(pdf_path: str):
    text = parse_pdf(pdf_path)
    metadata = extract_alas_metadata(text)
    
    client = MongoClient(os.getenv("MONGO_URI"))
    db = client["compliance"]["alas_docs"]
    db.insert_one(metadata)
    print("ALAS document parsed and stored in MongoDB.")
