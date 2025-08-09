import os
from datetime import timedelta

class Config:
    # JWT Configuration
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY') or 'your-secret-key-change-this-in-production'
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=24)
    
    # MongoDB Configuration
    MONGODB_URI = os.environ.get('MONGODB_URI') or "mongodb+srv://Darren:22p648@darren-robert.41bfjmg.mongodb.net/?retryWrites=true&w=majority&appName=darren-robert"
    DATABASE_NAME = 'stubotdb'
    
    # AI Model Configuration
    EMBEDDING_MODEL = "thenlper/gte-small"
    OLLAMA_MODEL = "gemma:2b"
    CHUNK_SIZE = 180
    CONTEXT_K = 10
    
    # Data Configuration
    STUDENTS_DATA_FILE = "data/Students Performance Dataset.json"
    
    # Server Configuration
    DEBUG = True
    HOST = '0.0.0.0'
    PORT = 5000