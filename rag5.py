from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import faiss
import subprocess
from sentence_transformers import SentenceTransformer
import language_tool_python
import threading
import time
from datetime import datetime
import sqlite3
from collections import defaultdict, Counter
import uuid
import re
from PyPDF2 import PdfReader

app = Flask(__name__)
CORS(app)  # Permet les requêtes depuis React

# Base de données pour sauvegarder les conversations
DB_PATH = 'conversations.db'

# Variables globales pour stocker le modèle et l'index
model = None
index = None
index2 = None
chunks = []
chunks2 = []
student_profiles = []
raw_students_data = []  # Données brutes pour les statistiques

# === Base de données pour les conversations ===
def init_database():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Table des conversations
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            session_id TEXT,
            user_message TEXT,
            bot_response TEXT,
            message_type TEXT,
            context_used INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Table des sessions
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            last_activity DATETIME DEFAULT CURRENT_TIMESTAMP,
            message_count INTEGER DEFAULT 0
        )
    ''')
    
    conn.commit()
    conn.close()

def save_conversation(session_id, user_message, bot_response, message_type='analysis', context_used=0):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Sauvegarder la conversation
    conversation_id = str(uuid.uuid4())
    cursor.execute('''
        INSERT INTO conversations (id, session_id, user_message, bot_response, message_type, context_used)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (conversation_id, session_id, user_message, bot_response, message_type, context_used))
    
    # Mettre à jour la session
    cursor.execute('''
        INSERT OR REPLACE INTO sessions (session_id, last_activity, message_count)
        VALUES (?, CURRENT_TIMESTAMP, 
                COALESCE((SELECT message_count FROM sessions WHERE session_id = ?), 0) + 1)
    ''', (session_id, session_id))
    
    conn.commit()
    conn.close()

def get_conversation_history(session_id, limit=50):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT user_message, bot_response, message_type, context_used, timestamp
        FROM conversations
        WHERE session_id = ?
        ORDER BY timestamp ASC
        LIMIT ?
    ''', (session_id, limit))
    
    results = cursor.fetchall()
    conn.close()
    return results

def get_analytics_data():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Statistiques générales
    cursor.execute('SELECT COUNT(*) FROM conversations')
    total_conversations = cursor.fetchone()[0]
    
    cursor.execute('SELECT COUNT(DISTINCT session_id) FROM sessions')
    total_sessions = cursor.fetchone()[0]
    
    # Messages par type
    cursor.execute('''
        SELECT message_type, COUNT(*) 
        FROM conversations 
        GROUP BY message_type
    ''')
    message_types = dict(cursor.fetchall())
    
    # Activité par jour (7 derniers jours)
    cursor.execute('''
        SELECT DATE(timestamp) as date, COUNT(*) as count
        FROM conversations
        WHERE timestamp >= date('now', '-7 days')
        GROUP BY DATE(timestamp)
        ORDER BY date
    ''')
    daily_activity = cursor.fetchall()
    
    conn.close()
    
    return {
        'total_conversations': total_conversations,
        'total_sessions': total_sessions,
        'message_types': message_types,
        'daily_activity': daily_activity
    }

# === Fonctions de votre code original ===
def load_students_from_json(json_file):
    global raw_students_data
    students = []
    raw_students_data = []
    
    with open(json_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                student = json.loads(line.strip())
                raw_students_data.append(student)  # Garder les données brutes
                
                # Construire une phrase narrative
                desc = (
                    f"{student['First_Name']} {student['Last_Name']} ({student['Gender']}, {student['Age']} ans), "
                    f"du département {student['Department']}, a un taux de présence de {student['Attendance (%)']}%, "
                    f"une note finale de {student['Final_Score']}/100, une participation de {student['Participation_Score']}/100, "
                    f"et un niveau de stress de {student['Stress_Level (1-10)']}/10. "
                    f"Étudie {student['Study_Hours_per_Week']} heures/semaine, dort {student['Sleep_Hours_per_Night']}h/nuit. "
                    f"Activités extrascolaires : {student['Extracurricular_Activities']}. "
                    f"Accès internet : {student['Internet_Access_at_Home']}. "
                    f"Note globale : {student['Grade']}."
                )
                students.append(desc)
            except Exception as e:
                print(f"❌ Ligne ignorée (erreur : {e})")
    return students
    # 1. Charger les documents physiques (PDF ici)
def load_pdf_documents(pdf_folder):
   
    if not os.path.exists(pdf_folder):
        print(f"Dossier PDF non trouvé: {pdf_folder}")
        return []
    
    documents = []
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
    
    if not pdf_files:
        print(f"Aucun fichier PDF trouvé dans {pdf_folder}")
        return documents
    
    for filename in pdf_files:
        try:
            file_path = os.path.join(pdf_folder, filename)
            print(f"Traitement: {filename}")
            
            reader = PdfReader(file_path)
            text_parts = []
            
            for page in reader.pages:
                try:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text_parts.append(page_text.strip())
                except Exception as e:
                    print(f"Erreur page dans {filename}: {e}")
                    continue
            
            if text_parts:
                full_text = "\n".join(text_parts)
                documents.append(full_text)
                print(f"✅ {filename}: {len(text_parts)} pages")
            else:
                print(f"❌ Aucun texte extrait de {filename}")
                
        except Exception as e:
            print(f"Erreur traitement {filename}: {e}")
            continue
    
    print(f"✅ {len(documents)} documents PDF chargés")
    return documents

def split_text(texts, chunk_size=500):
    
    chunks = []
    for text in texts:
        if not text or not text.strip():
            continue
        words = text.split()
        chunks.extend([" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size) if len(words[i:i+chunk_size]) > 10])
    return chunks

def create_faiss_index(chunks, model):
    if not chunks:
        print
    vectors = model.encode(chunks)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index, vectors
def create_faiss_index2(chunks, model):

    if not chunks:
        raise ValueError("Aucun chunk fourni pour l'indexation")
    
    vectors = model.encode(chunks, show_progress_bar=True)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors.astype('float32'))
    return index, vectors

def retrieve_context(question, chunks, model, index, k=5):
    question_vec = model.encode([question])
    _, I = index.search(question_vec, k)
    return "\n".join([chunks[i] for i in I[0]])

def generate_with_ollama(prompt, model_name="gemma:2b"):
    try:
        result = subprocess.run(
            ["ollama", "run", model_name],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300  # Timeout de 5 minutes pour éviter les blocages
        )
        return result.stdout.decode("utf-8").strip()
    except subprocess.TimeoutExpired:
        return "Désolé, la génération a pris trop de temps."
    except Exception as e:
        return f"Erreur lors de la génération : {str(e)}"

def postprocess_response(response: str) -> str:
    try:
        tool = language_tool_python.LanguageTool('fr-FR')
        matches = tool.check(response)
        corrected = language_tool_python.utils.correct(response, matches)
        return corrected
    except Exception as e:
        print(f"⚠️ Erreur pendant la correction locale : {e}")
        return response
def is_nonsense(message):
    message = message.strip().lower()
    
    # Trop court et pas pertinent
    if len(message) < 4:
        return True

    # Mot sans voyelle ou répété
    if not re.search(r'[aeiouy]', message):
        return True

    # Répétition de même lettre ou pattern du type "hhh", "aaaaa", "lololol"
    if re.fullmatch(r'(.)\1{2,}', message):  # 3 lettres ou plus identiques
        return True

    if re.fullmatch(r'([a-z]{1,2})\1{2,}', message):  # ex: "haha", "lololol"
        return True

    return False
# === Initialisation du système ===
def initialize_system():
    global model, index, chunks, student_profiles
    
    try:
        print("🔍 Initialisation du système...")
        json_file = "data/Students Performance Dataset.json"
        PDF="data/data"
       
        # Initialiser la base de données
        init_database()
        
        
        print("🔍 Chargement des documents...")
        Docs = load_pdf_documents(PDF)
        

        print("🔍 Chargement des profils étudiants...")
        student_profiles = load_students_from_json(json_file)
        
        print("🔍 Découpage des descriptions en chunks...")
        chunks = split_text(student_profiles)
        chunks2 = split_text(Docs)
        
        
        print("🔍 Chargement du modèle d'embeddings...")
        model = SentenceTransformer("thenlper/gte-small")
       
        
        print("🔍 Création de l'index FAISS...")
        index, _ = create_faiss_index(chunks, model)
        if chunks2:
            print("🔍 Création de l'index FAISS pour documents...")
            index2, _ = create_faiss_index2(chunks2, model)
        else:
            print("ℹ️ Pas de documents PDF, index2 non créé")
            index2 = None
        
        
        print("✅ Système initialisé avec succès!")
        return True
    except Exception as e:
        print(f"❌ Erreur lors de l'initialisation : {e}")
        return False

# === Routes API ===
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "index_loaded": index is not None,
        "index2_loaded": index2 is not None,
        "chunks_count": len(chunks),
        "chunks2_count": len(chunks2)
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    global model, index, chunks, index2, chunks2,student_profiles
    
    if not all([model, index, chunks, index2, chunks2,]):
        return jsonify({
            "error": "Système non initialisé",
            "message": "Veuillez attendre l'initialisation du système"
        }), 503
    
    try:
        data = request.get_json()
        question = data.get('message', '').strip()
        session_id = data.get('session_id', str(uuid.uuid4()))

        if not question:
            return jsonify({"error": "Message vide"}), 400
        if is_nonsense(question):
            response = "🤔 Je n’ai pas compris votre message. Pouvez-vous reformuler ou poser une question plus claire ?"
            save_conversation(session_id, question, response, 'nonsense', 0)
            return jsonify({
                "response": response,
                "type": "nonsense",
                "session_id": session_id
            })
        # Gestion des smalltalk et questions générales
        smalltalk = ["bonjour", "salut", "merci", "ça va", "au revoir", "hello"]
        general_questions = [
            "quel est ton rôle", "tu fais quoi", "qui es-tu", "c'est quoi ton travail",
            "tu peux m'aider", "que fais-tu", "tu sers à quoi", "tu es qui"
        ]
        
        q_lower = question.lower()
        
        if any(word in q_lower for word in smalltalk):
            response = "👋 Salut ! Je suis ton assistant éducatif. Pose-moi une question sur un étudiant si tu veux une analyse 😊."
            save_conversation(session_id, question, response, 'greeting', 0)
            return jsonify({
                "response": response,
                "type": "greeting",
                "session_id": session_id
            })
        
        if any(phrase in q_lower for phrase in general_questions):
            response = "🤖 Je suis un assistant éducatif intelligent. Mon rôle est d'analyser et prédire le comportement des étudiants pour identifier ceux à risque et proposer des solutions 👍."
            save_conversation(session_id, question, response, 'info', 0)
            return jsonify({
                "response": response,
                "type": "info",
                "session_id": session_id
            })
        
        # Analyse normale
        context = retrieve_context(question, chunks, model, index)
        context_count = len(context.split('\n'))
        context2 = retrieve_context(question, chunks2, index2)
        
        prompt = f"""
Tu es un conseiller pédagogique spécialisé en analyse et prédiction comportementale des étudiants.

== Profils étudiants ==
{context}

Question de l'utilisateur :
{question}
== Informations issues de documents scientifiques ==
{context2}

Réponds de manière structurée, pédagogique et concise.

Réponse :
"""
        
        raw_response = generate_with_ollama(prompt)
        final_response = postprocess_response(raw_response)
        
        # Sauvegarder la conversation
        save_conversation(session_id, question, final_response, 'analysis', context_count)
        
        return jsonify({
            "response": final_response,
            "type": "analysis",
            "context_used": context_count,
            "session_id": session_id
        })
        
    except Exception as e:
        return jsonify({
            "error": "Erreur lors du traitement",
            "message": str(e)
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    return jsonify({
        "total_students": len(student_profiles),
        "total_chunks": len(chunks),
        "model_name": "thenlper/gte-small",
        "system_status": "operational" if all([model, index, chunks]) else "initializing"
    })

# === Nouvelles routes pour les fonctionnalités avancées ===

@app.route('/api/conversations/<session_id>', methods=['GET'])
def get_conversations(session_id):
    try:
        history = get_conversation_history(session_id)
        formatted_history = []
        
        for user_msg, bot_response, msg_type, context_used, timestamp in history:
            formatted_history.extend([
                {
                    "type": "user",
                    "content": user_msg,
                    "timestamp": timestamp
                },
                {
                    "type": "bot",
                    "content": bot_response,
                    "message_type": msg_type,
                    "context_used": context_used,
                    "timestamp": timestamp
                }
            ])
        
        return jsonify({"history": formatted_history})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    try:
        analytics = get_analytics_data()
        return jsonify(analytics)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/student-analytics', methods=['GET'])
def get_student_analytics():
    try:
        if not raw_students_data:
            return jsonify({"error": "Données étudiants non chargées"}), 503
        
        # Analyse des données étudiantes
        departments = Counter(student['Department'] for student in raw_students_data)
        grades = Counter(student['Grade'] for student in raw_students_data)
        gender_distribution = Counter(student['Gender'] for student in raw_students_data)
        
        # Statistiques de performance
        scores = [student['Final_Score'] for student in raw_students_data]
        attendance = [student['Attendance (%)'] for student in raw_students_data]
        stress_levels = [student['Stress_Level (1-10)'] for student in raw_students_data]
        
        # Corrélations intéressantes
        performance_data = []
        for student in raw_students_data:
            performance_data.append({
                'name': f"{student['First_Name']} {student['Last_Name']}",
                'attendance': student['Attendance (%)'],
                'final_score': student['Final_Score'],
                'stress_level': student['Stress_Level (1-10)'],
                'study_hours': student['Study_Hours_per_Week'],
                'sleep_hours': student['Sleep_Hours_per_Night'],
                'department': student['Department'],
                'grade': student['Grade']
            })
        
        # Étudiants à risque (score < 60 ou présence < 70%)
        at_risk_students = [
            student for student in raw_students_data
            if student['Final_Score'] < 60 or student['Attendance (%)'] < 70
        ]
        
        return jsonify({
            'departments': dict(departments),
            'grades': dict(grades),
            'gender_distribution': dict(gender_distribution),
            'performance_stats': {
                'avg_score': sum(scores) / len(scores),
                'avg_attendance': sum(attendance) / len(attendance),
                'avg_stress': sum(stress_levels) / len(stress_levels)
            },
            'performance_data': performance_data,
            'at_risk_count': len(at_risk_students),
            'total_students': len(raw_students_data)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Initialisation en arrière-plan
def init_background():
    initialize_system()

if __name__ == '__main__':
    # Lancer l'initialisation en arrière-plan
    init_thread = threading.Thread(target=init_background)
    init_thread.daemon = True
    init_thread.start()
    
    print("🚀 Démarrage du serveur Flask...")
    print("🔧 Initialisation en cours en arrière-plan...")
    
    app.run( host='0.0.0.0', port=5000)