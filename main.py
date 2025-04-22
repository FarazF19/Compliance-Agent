from loaders.load_pdf import load_eu_act_and_store
import os
from loaders.parse_alas import store_alas_to_mongo
    

if __name__ == "__main__":
    # pdf_path = "docs/EU_AI_Act.pdf"  
    # index_name = "eu-ai-act-index"


    # if not os.getenv("OPENAI_API_KEY") or not os.getenv("PINECONE_API_KEY") or not os.getenv("PINECONE_ENV"):
    #     raise EnvironmentError("Missing required environment variables. Check your .env file.")

    # Calling ingestion pipeline
    # load_eu_act_and_store(pdf_path=pdf_path, index_name=index_name)
    
    # Path to ALAS sample use case
    pdf_path = "docs/alas_sample_use_case.pdf"
    
    # Call the function to parse and store metadata
    store_alas_to_mongo(pdf_path)
