# chatbot-intelligent-pour-la-preeiction-et-l-analyse-du-comportement
premiere verision
🎓 Student Analysis backend
Structure du projet modulaire
project/
├── app.py                      # Application principale (point d'entrée)
├── config.py                   # Configuration centralisée
├── database.py                 # Gestion MongoDB
├── requirements.txt            # Dépendances Python
├── .env                        # Variables d'environnement (à créer)
├── routes/
│   ├── __init__.py
│   ├── auth.py                 # Routes d'authentification
│   ├── chat.py                 # Routes de chat et analyse
│   └── admin.py                # Routes administrateur
├── services/
│   ├── __init__.py
│   └── ai_service.py           # Service IA (Ollama, FAISS, etc.)
└── data/
    └── Students Performance Dataset.json
Installer les dépendances
pip install -r requirements.txt
Démarrer l'application
python app.py
