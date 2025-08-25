from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from collections import Counter
import uuid
from services.ai_service import ai_service
from database import db_instance

class ChatController:
    """Contrôleur pour la gestion du chat et des conversations"""
    
    def __init__(self, ai_service, db_instance):
        self.ai_service = ai_service
        self.db = db_instance
        
        # Configuration des mots-clés pour les différents types de messages
        self.smalltalk_keywords = ["bonjour", "salut", "merci", "ça va", "au revoir", "hello"]
        self.general_questions = [
            "quel est ton rôle", "tu fais quoi", "qui es-tu", "c'est quoi ton travail",
            "tu peux m'aider", "que fais-tu", "tu sers à quoi", "tu es qui"
        ]
        
       
    
    def health_check(self):
        """Vérification de l'état de santé du système"""
        return jsonify({
            "status": "healthy",
            "model_loaded": self.ai_service.model is not None,
            "index_loaded": self.ai_service.index is not None,
            "chunks_count": len(self.ai_service.chunks),
            "mongodb_connected": self.db.db is not None
        })
    
    def chat(self):
        """Traitement des messages de chat"""
        if not self.ai_service.is_initialized():
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
            
            # Vérifier si le message est nonsense
            if self.ai_service.is_nonsense(question):
                response = "🤔 Je n'ai pas compris votre message. Pouvez-vous reformuler ou poser une question plus claire ?"
                self.db.save_conversation(session_id, question, response, 'nonsense', 0, user_id)
                return jsonify({
                    "response": response,
                    "type": "nonsense",
                    "session_id": session_id
                })

            q_lower = question.lower()
            
            # Gestion des salutations et smalltalk
            if self._is_smalltalk(q_lower):
                response = "👋 Salut ! Je suis ton assistant éducatif. Pose-moi une question sur un étudiant si tu veux une analyse 😊."
                self.db.save_conversation(session_id, question, response, 'greeting', 0, user_id)
                return jsonify({
                    "response": response,
                    "type": "greeting",
                    "session_id": session_id
                })
            
            # Gestion des questions générales sur le rôle
            if self._is_general_question(q_lower):
                response = "🤖 Je suis un assistant éducatif intelligent. Mon rôle est d'analyser et prédire le comportement des étudiants pour identifier ceux à risque et proposer des solutions 👍."
                self.db.save_conversation(session_id, question, response, 'info', 0, user_id)
                return jsonify({
                    "response": response,
                    "type": "info",
                    "session_id": session_id
                })
            
            # Analyse normale avec IA
            return self._process_ai_analysis(question, session_id, user_id)
            
        except Exception as e:
            return jsonify({
                "error": "Erreur lors du traitement",
                "message": str(e)
            }), 500
    
    def get_conversations(self, session_id):
        """Récupération de l'historique des conversations"""
        try:
            user_id = get_jwt_identity()
            history = self.db.get_conversation_history(session_id)
            
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
                        "similarity_score": conv.get('similarity_score', 0),
                        "timestamp": conv['timestamp'].isoformat()
                    }
                ])
            
            return jsonify({"history": formatted_history})
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    def _is_smalltalk(self, question_lower):
        """Vérifie si la question est du smalltalk"""
        return any(word in question_lower for word in self.smalltalk_keywords)
    
    def _is_general_question(self, question_lower):
        """Vérifie si c'est une question générale sur le rôle"""
        return any(phrase in question_lower for phrase in self.general_questions)
    
   
    
    def _process_ai_analysis(self, question, session_id, user_id):
        """Traite une question avec l'analyse IA"""
        # Récupération du contexte avec score de similarité
        context, similarity_score = self.ai_service.retrieve_context(question)
        
        context_count = len(context.split('\n'))
        
        
        # Prompt standard pour l'analyse générale
        prompt = f"""Tu es StuBot, un conseiller pédagogique expert, spécialisé dans l'analyse de profils étudiants. Ta mission est d'aider les éducateurs à comprendre les comportements des étudiants et à proposer des plans d'action ciblés.

Règles à suivre impérativement :
- Tu ne dois faire aucune supposition ou inférence qui ne soit pas explicitement présente dans les données fournies.
- Ton analyse doit se baser uniquement sur les "extraits de profils d'étudiants" et la "question de l'utilisateur".

Voici la question de l'utilisateur :
{question}

Pour ton analyse, voici des extraits de profils d'étudiants aux caractéristiques similaires :
{context}

En te basant strictement sur ces informations, génère une réponse structurée en trois sections distinctes :

### 1. Analyse des facteurs de succès et de risque
Identifie les facteurs clés (présence, notes, participation, stress, sommeil, etc.) qui semblent corrélés avec le succès académique ou les difficultés rencontrées, d'après les profils fournis. Fais le lien entre ces facteurs de manière factuelle.

### 2. Aperçu des tendances comportementales
Décris les tendances comportementales communes que tu observes dans les profils qui réussissent par rapport à ceux qui rencontrent des difficultés. Par exemple, comment la moyenne des heures de sommeil se compare-t-elle entre les groupes ?

### 3. Recommandations et plan d'action
Propose des recommandations concrètes, realistes et spécifiques. Pour chaque recommandation, associe-la à un ou plusieurs facteurs identifiés dans ton analyse. Organise tes conseils pour qu'ils soient directement applicables par un éducateur (par exemple, "Encourager les étudiants avec un faible taux de présence à consulter les notes de cours en ligne").

Réponse :
"""
        
        # Génération de la réponse
        # --- ANCIEN APPEL AVEC OLLAMA ---
        # raw_response = self.ai_service.generate_with_ollama(prompt)
        
        # --- NOUVEL APPEL AVEC GEMINI ---
        raw_response = self.ai_service.generate_with_openrouter(prompt)
        final_response = self.ai_service.postprocess_response(raw_response)
        
        # Sauvegarde de la conversation avec le score de similarité
        self.db.save_conversation(session_id, question, final_response, context_count, user_id, similarity_score)
        
        return jsonify({
            "response": final_response,
            "type": 'analysis',
            "context_used": context_count,
            "similarity_score": round(similarity_score, 3),
            "session_id": session_id
        })

# Créer une instance du contrôleur
chat_controller = ChatController(ai_service, db_instance)

# Créer le blueprint
chat_bp = Blueprint('chat', __name__)

# Enregistrer les routes avec les méthodes de la classe
chat_bp.route('/health', methods=['GET'])(chat_controller.health_check)
chat_bp.route('/chat', methods=['POST'])(jwt_required()(chat_controller.chat))
chat_bp.route('/conversations/<session_id>', methods=['GET'])(jwt_required()(chat_controller.get_conversations))