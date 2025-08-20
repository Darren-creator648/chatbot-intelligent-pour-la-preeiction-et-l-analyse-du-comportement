import os
from datetime import timedelta
from dotenv import load_dotenv

load_dotenv() # Charge les variables d'environnement du fichier .env

class Config:
    # JWT Configuration
    JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY') or 'your-secret-key-change-this-in-production'
    JWT_ACCESS_TOKEN_EXPIRES = timedelta(hours=24)
    
    # MongoDB Configuration
    MONGODB_URI = os.environ.get('MONGODB_URI') or "mongodb+srv://Darren:22p648@darren-robert.41bfjmg.mongodb.net/?retryWrites=true&w=majority&appName=darren-robert"
    DATABASE_NAME = 'stubotdb'
    
    # AI Model Configuration
    EMBEDDING_MODEL = "thenlper/gte-small"
    #OLLAMA_MODEL = "gemma:2b"
    CHUNK_SIZE = 80
    CONTEXT_K = 5
    # Nouveau modèle Gemini Pro
    
    #GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY') # Meilleure pratique : stocker la clé en tant que variable d'environnement
    #GEMINI_MODEL = "google/gemini-2.5-pro-exp-03-25"  # Nom du modèle sur l'API de Google
    # Nouvelle configuration OpenRouter
    OPENROUTER_API_KEY = os.environ.get('OPENROUTER_API_KEY')
    OPENROUTER_MODEL = "mistralai/mistral-7b-instruct:free"
    
    # Data Configuration
    STUDENTS_DATA_FILE = "data/Students Performance Dataset.json"
    STUDENTS_DATA_URL = "data/Students Performance Datasete.json"
    
    # Server Configuration
    DEBUG = True
    HOST = '0.0.0.0'
    PORT = 5000