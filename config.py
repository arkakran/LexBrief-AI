from pydantic_settings import BaseSettings
from pydantic import Field
import os

class Settings(BaseSettings):
    # API Keys
    groq_api_key: str
    qdrant_url: str
    qdrant_api_key: str
    huggingface_api_key: str
    
    # Models
    llm_model: str = "llama-3.3-70b-versatile"
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    
    # Processing
    chunk_size: int = 1000
    chunk_overlap: int = 200
    top_k_retrieval: int = 60  
    top_k_reranked: int = 25   
    final_output_count: int = 10
    
    # App
    upload_folder: str = "data/uploads"
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        os.makedirs(self.upload_folder, exist_ok=True)

settings = Settings()
