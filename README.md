# chatbot-intelligent-pour-la-preeiction-et-l-analyse-du-comportement
🎓 Student Analysis Frontend
Application React pour l'analyse comportementale des étudiants avec assistant IA intégré.
📋 Table des matières

Prérequis
Installation
Configuration
Démarrage

🔧 Prérequis
Avant de commencer, assurez-vous d'avoir installé :

Node.js (version 16.0.0 ou supérieure)
npm ou yarn
Git

Vérifiez vos versions :
bashnode --version
npm --version
🚀 Installation
1. Cloner le projet (ou créer un nouveau projet)
bash# Option 1: Cloner le repository existant
git clone https://github.com/Darren-creator648/chatbot-intelligent-pour-la-preeiction-et-l-analyse-du-comportement
cd student-analysis-frontende

# Option 2: Créer un nouveau projet React
npx create-react-app student-analysis-frontend
cd student-analysis-frontend
2. Installer les dépendances
bash# Installer les packages nécessaires
npm install lucide-react recharts

# Ou avec yarn
yarn add lucide-react recharts
3. Configurer Tailwind CSS
bash# Installer Tailwind CSS et ses dépendances
npm install -D tailwindcss@3
npx tailwindcss init

# Initialiser Tailwind
npx tailwindcss init -p
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
