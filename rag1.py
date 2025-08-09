from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity
from werkzeug.security import generate_password_hash, check_password_hash
from pymongo import MongoClient
from datetime import datetime, timedelta
import os
import json
import faiss
import subprocess
from sentence_transformers import SentenceTransformer
import language_tool_python
import threading
import time
from collections import defaultdict, Counter
import uuid
import re
from bson import ObjectId

app = Flask(__name__)
CORS(app)

# Configuration JWT
app.config['JWT_SECRET_KEY'] = 'your-secret-key-change-this-in-production'  # Changez cette clé en production
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)
jwt = JWTManager(app)

# Configuration MongoDB
MONGODB_URI = "mongodb+srv://Darren:22p648@darren-robert.41bfjmg.mongodb.net/?retryWrites=true&w=majority&appName=darren-robert"
DATABASE_NAME = 'stubotdb'

# Connexion MongoDB
try:
    client = MongoClient(MONGODB_URI)
    db = client[DATABASE_NAME]
    
    # Collections
    users_collection = db.users
    conversations_collection = db.conversations
    sessions_collection = db.sessions
    
    print("✅ Connexion MongoDB établie")
except Exception as e:
    print(f"❌ Erreur connexion MongoDB: {e}")
    db = None

# Variables globales pour stocker le modèle et l'index
model = None
index = None
chunks = []
student_profiles = []
raw_students_data = []

# === Fonctions utilitaires MongoDB ===
def serialize_mongo_document(doc):
    """Convertit les ObjectId MongoDB en string pour JSON"""
    if doc is None:
        return None
    if isinstance(doc, list):
        return [serialize_mongo_document(item) for item in doc]
    if isinstance(doc, dict):
        for key, value in doc.items():
            if isinstance(value, ObjectId):
                doc[key] = str(value)
            elif isinstance(value, datetime):
                doc[key] = value.isoformat()
            elif isinstance(value, dict):
                doc[key] = serialize_mongo_document(value)
            elif isinstance(value, list):
                doc[key] = serialize_mongo_document(value)
    return doc

# === Authentification ===
@app.route('/api/register', methods=['POST'])
def register():
    try:
        data = request.get_json()
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        full_name = data.get('full_name', '').strip()
        role = data.get('role', 'student')  # student, teacher, admin
        
        # Validation
        if not email or not password or not full_name:
            return jsonify({'error': 'Tous les champs sont requis'}), 400
        
        if len(password) < 6:
            return jsonify({'error': 'Le mot de passe doit contenir au moins 6 caractères'}), 400
        
        # Vérifier si l'utilisateur existe
        if users_collection.find_one({'email': email}):
            return jsonify({'error': 'Cet email est déjà utilisé'}), 400
        
        # Créer l'utilisateur
        hashed_password = generate_password_hash(password)
        user_data = {
            'email': email,
            'password': hashed_password,
            'full_name': full_name,
            'role': role,
            'created_at': datetime.utcnow(),
            'last_login': None,
            'is_active': True
        }
        
        result = users_collection.insert_one(user_data)
        
        # Créer le token JWT
        access_token = create_access_token(
            identity=str(result.inserted_id),
            additional_claims={'email': email, 'role': role}
        )
        
        return jsonify({
            'message': 'Compte créé avec succès',
            'access_token': access_token,
            'user': {
                'id': str(result.inserted_id),
                'email': email,
                'full_name': full_name,
                'role': role
            }
        }), 201
        
    except Exception as e:
        return jsonify({'error': f'Erreur serveur: {str(e)}'}), 500

@app.route('/api/login', methods=['POST'])
def login():
    try:
        data = request.get_json()
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        
        if not email or not password:
            return jsonify({'error': 'Email et mot de passe requis'}), 400
        
        # Trouver l'utilisateur
        user = users_collection.find_one({'email': email})
        if not user or not check_password_hash(user['password'], password):
            return jsonify({'error': 'Email ou mot de passe incorrect'}), 401
        
        if not user.get('is_active', True):
            return jsonify({'error': 'Compte désactivé'}), 401
        
        # Mettre à jour la dernière connexion
        users_collection.update_one(
            {'_id': user['_id']},
            {'$set': {'last_login': datetime.utcnow()}}
        )
        
        # Créer le token JWT
        access_token = create_access_token(
            identity=str(user['_id']),
            additional_claims={'email': email, 'role': user.get('role', 'student')}
        )
        
        return jsonify({
            'message': 'Connexion réussie',
            'access_token': access_token,
            'user': {
                'id': str(user['_id']),
                'email': user['email'],
                'full_name': user['full_name'],
                'role': user.get('role', 'student')
            }
        })
        
    except Exception as e:
        return jsonify({'error': f'Erreur serveur: {str(e)}'}), 500

@app.route('/api/profile', methods=['GET'])
@jwt_required()
def get_profile():
    try:
        user_id = get_jwt_identity()
        user = users_collection.find_one({'_id': ObjectId(user_id)})
        
        if not user:
            return jsonify({'error': 'Utilisateur non trouvé'}), 404
        
        user_data = serialize_mongo_document(user)
        # Supprimer le mot de passe
        user_data.pop('password', None)
        
        return jsonify({'user': user_data})
        
    except Exception as e:
        return jsonify({'error': f'Erreur serveur: {str(e)}'}), 500

# === Gestion des conversations avec MongoDB ===
def save_conversation(session_id, user_message, bot_response, message_type='analysis', context_used=0, user_id=None):
    try:
        conversation_data = {
            'conversation_id': str(uuid.uuid4()),
            'session_id': session_id,
            'user_id': user_id,
            'user_message': user_message,
            'bot_response': bot_response,
            'message_type': message_type,
            'context_used': context_used,
            'timestamp': datetime.utcnow()
        }
        
        conversations_collection.insert_one(conversation_data)
        
        # Mettre à jour ou créer la session
        sessions_collection.update_one(
            {'session_id': session_id},
            {
                '$set': {
                    'last_activity': datetime.utcnow(),
                    'user_id': user_id
                },
                '$inc': {'message_count': 1},
                '$setOnInsert': {'created_at': datetime.utcnow()}
            },
            upsert=True
        )
        
    except Exception as e:
        print(f"Erreur sauvegarde conversation: {e}")

def get_conversation_history(session_id, limit=50):
    try:
        conversations = conversations_collection.find(
            {'session_id': session_id}
        ).sort('timestamp', 1).limit(limit)
        
        return list(conversations)
    except Exception as e:
        print(f"Erreur récupération historique: {e}")
        return []

def get_analytics_data(user_id=None):
    try:
        # Filtrer par utilisateur si spécifié
        filter_query = {'user_id': user_id} if user_id else {}
        
        # Statistiques générales
        total_conversations = conversations_collection.count_documents(filter_query)
        total_sessions = sessions_collection.count_documents(filter_query)
        
        # Messages par type
        pipeline = [
            {'$match': filter_query},
            {'$group': {'_id': '$message_type', 'count': {'$sum': 1}}}
        ]
        message_types_cursor = conversations_collection.aggregate(pipeline)
        message_types = {item['_id']: item['count'] for item in message_types_cursor}
        
        # Activité par jour (7 derniers jours)
        seven_days_ago = datetime.utcnow() - timedelta(days=7)
        pipeline = [
            {'$match': {**filter_query, 'timestamp': {'$gte': seven_days_ago}}},
            {
                '$group': {
                    '_id': {'$dateToString': {'format': '%Y-%m-%d', 'date': '$timestamp'}},
                    'count': {'$sum': 1}
                }
            },
            {'$sort': {'_id': 1}}
        ]
        daily_activity_cursor = conversations_collection.aggregate(pipeline)
        daily_activity = [(item['_id'], item['count']) for item in daily_activity_cursor]
        
        return {
            'total_conversations': total_conversations,
            'total_sessions': total_sessions,
            'message_types': message_types,
            'daily_activity': daily_activity
        }
    except Exception as e:
        print(f"Erreur analytics: {e}")
        return {}

# === Fonctions de votre code original (inchangées) ===
def load_students_from_json(json_file):
    global raw_students_data
    students = []
    raw_students_data = []
    
    with open(json_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                student = json.loads(line.strip())
                raw_students_data.append(student)
                
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
# === Découpage du texte en chunks ===
# Cette fonction découpe les textes en chunks de taille fixe (par défaut 80 mots)

def split_text(texts, chunk_size=80):
    chunks = []
    for text in texts:
        words = text.split()
        chunks.extend([" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)])
    return chunks
# === Création de l'index FAISS ===
def create_faiss_index(chunks, model):
    vectors = model.encode(chunks)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index, vectors
# === Fonctions de récupération de contexte  ===
# Cette fonction récupère le contexte pertinent pour une question donnée en utilisant l'index FAISS
# Elle renvoie les k chunks les plus pertinents.
def retrieve_context(question, chunks, model, index, k=5):
    question_vec = model.encode([question])
    _, I = index.search(question_vec, k)
    return "\n".join([chunks[i] for i in I[0]])
# === Génération de réponse avec Ollama ===
def generate_with_ollama(prompt, model_name="gemma:2b"):
    try:
        result = subprocess.run(
            ["ollama", "run", model_name],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300
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
    if len(message) < 4:
        return True
    if not re.search(r'[aeiouy]', message):
        return True
    if re.fullmatch(r'(.)\1{2,}', message):
        return True
    if re.fullmatch(r'([a-z]{1,2})\1{2,}', message):
        return True
    return False

def initialize_system():
    global model, index, chunks, student_profiles
    
    try:
        print("🔍 Initialisation du système...")
        json_file = "data/Students Performance Dataset.json"
        
        print("🔍 Chargement des profils étudiants...")
        student_profiles = load_students_from_json(json_file)
        
        print("🔍 Découpage des descriptions en chunks...")
        chunks = split_text(student_profiles)
        
        print("🔍 Chargement du modèle d'embeddings...")
        model = SentenceTransformer("thenlper/gte-small")
        
        print("🔍 Création de l'index FAISS...")
        index, _ = create_faiss_index(chunks, model)
        
        print("✅ Système initialisé avec succès!")
        return True
    except Exception as e:
        print(f"❌ Erreur lors de l'initialisation : {e}")
        return False

# === Routes API (mises à jour pour l'authentification) ===
@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "index_loaded": index is not None,
        "chunks_count": len(chunks),
        "mongodb_connected": db is not None
    })

@app.route('/api/chat', methods=['POST'])
@jwt_required()
def chat():
    global model, index, chunks
    
    if not all([model, index, chunks]):
        return jsonify({
            "error": "Système non initialisé",
            "message": "Veuillez attendre l'initialisation du système"
        }), 503
    
    try:
        user_id = get_jwt_identity()
        data = request.get_json()
        question = data.get('message', '').strip()
        session_id = data.get('session_id', str(uuid.uuid4()))

        if not question:
            return jsonify({"error": "Message vide"}), 400
        
        if is_nonsense(question):
            response = "🤔 Je n'ai pas compris votre message. Pouvez-vous reformuler ou poser une question plus claire ?"
            save_conversation(session_id, question, response, 'nonsense', 0, user_id)
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
            save_conversation(session_id, question, response, 'greeting', 0, user_id)
            return jsonify({
                "response": response,
                "type": "greeting",
                "session_id": session_id
            })
        
        if any(phrase in q_lower for phrase in general_questions):
            response = "🤖 Je suis un assistant éducatif intelligent. Mon rôle est d'analyser et prédire le comportement des étudiants pour identifier ceux à risque et proposer des solutions 👍."
            save_conversation(session_id, question, response, 'info', 0, user_id)
            return jsonify({
                "response": response,
                "type": "info",
                "session_id": session_id
            })
        
        # Analyse normale
        context = retrieve_context(question, chunks, model, index)
        context_count = len(context.split('\n'))
        
        prompt = f"""
Tu es un conseiller pédagogique spécialisé en analyse et prédiction comportementale des étudiants.

Voici des extraits de profils d'étudiants :
{context}

Question de l'utilisateur :
{question}

Réponds de manière structurée, pédagogique et concise.

Réponse :
"""
        
        raw_response = generate_with_ollama(prompt)
        final_response = postprocess_response(raw_response)
        
        # Sauvegarder la conversation
        save_conversation(session_id, question, final_response, 'analysis', context_count, user_id)
        
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
@jwt_required()
def get_stats():
    return jsonify({
        "total_students": len(student_profiles),
        "total_chunks": len(chunks),
        "model_name": "thenlper/gte-small",
        "system_status": "operational" if all([model, index, chunks]) else "initializing"
    })

@app.route('/api/conversations/<session_id>', methods=['GET'])
@jwt_required()
def get_conversations(session_id):
    try:
        user_id = get_jwt_identity()
        history = get_conversation_history(session_id)
        
        # Filtrer les conversations de l'utilisateur actuel
        user_history = [conv for conv in history if conv.get('user_id') == user_id]
        
        formatted_history = []
        for conv in user_history:
            formatted_history.extend([
                {
                    "type": "user",
                    "content": conv['user_message'],
                    "timestamp": conv['timestamp'].isoformat()
                },
                {
                    "type": "bot",
                    "content": conv['bot_response'],
                    "message_type": conv['message_type'],
                    "context_used": conv['context_used'],
                    "timestamp": conv['timestamp'].isoformat()
                }
            ])
        
        return jsonify({"history": formatted_history})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/analytics', methods=['GET'])
@jwt_required()
def get_analytics():
    try:
        user_id = get_jwt_identity()
        analytics = get_analytics_data(user_id)
        return jsonify(analytics)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/student-analytics', methods=['GET'])
@jwt_required()
def get_student_analytics():
    try:
        if not raw_students_data:
            return jsonify({"error": "Données étudiants non chargées"}), 503
        
        # Analyse des données étudiantes (identique à l'original)
        departments = Counter(student['Department'] for student in raw_students_data)
        grades = Counter(student['Grade'] for student in raw_students_data)
        gender_distribution = Counter(student['Gender'] for student in raw_students_data)
        
        scores = [student['Final_Score'] for student in raw_students_data]
        attendance = [student['Attendance (%)'] for student in raw_students_data]
        stress_levels = [student['Stress_Level (1-10)'] for student in raw_students_data]
        
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

# Route admin pour obtenir tous les utilisateurs
@app.route('/api/admin/users', methods=['GET'])
@jwt_required()
def get_all_users():
    try:
        # Vérifier si l'utilisateur est admin (vous pouvez ajouter cette logique)
        users = users_collection.find({}, {'password': 0})  # Exclure les mots de passe
        users_list = [serialize_mongo_document(user) for user in users]
        return jsonify({'users': users_list})
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
    print("🔐 Authentification JWT activée")
    print("🗄️ Base de données MongoDB configurée")
    
    app.run(debug=True, host='0.0.0.0', port=5000)