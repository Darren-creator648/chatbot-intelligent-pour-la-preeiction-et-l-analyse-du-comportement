from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, get_jwt_identity
from collections import Counter
import uuid
from services.ai_service import ai_service
from database import db_instance

chat_bp = Blueprint('chat', __name__)

@chat_bp.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_loaded": ai_service.model is not None,
        "index_loaded": ai_service.index is not None,
        "chunks_count": len(ai_service.chunks),
        "mongodb_connected": db_instance.db is not None
    })

@chat_bp.route('/chat', methods=['POST'])
@jwt_required()
def chat():
    if not ai_service.is_initialized():
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
        
        if ai_service.is_nonsense(question):
            response = "🤔 Je n'ai pas compris votre message. Pouvez-vous reformuler ou poser une question plus claire ?"
            db_instance.save_conversation(session_id, question, response, 'nonsense', 0, user_id)
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
            db_instance.save_conversation(session_id, question, response, 'greeting', 0, user_id)
            return jsonify({
                "response": response,
                "type": "greeting",
                "session_id": session_id
            })
        
        if any(phrase in q_lower for phrase in general_questions):
            response = "🤖 Je suis un assistant éducatif intelligent. Mon rôle est d'analyser et prédire le comportement des étudiants pour identifier ceux à risque et proposer des solutions 👍."
            db_instance.save_conversation(session_id, question, response, 'info', 0, user_id)
            return jsonify({
                "response": response,
                "type": "info",
                "session_id": session_id
            })
        
        # Analyse normale
        context = ai_service.retrieve_context(question)
        context_count = len(context.split('\n'))
        
        prompt = f"""
Tu es un conseiller pédagogique spécialisé en analyse et prédiction comportementale des étudiants.

Voici des extraits de profils d'étudiants :
{context}

Question de l'utilisateur :
{question}

Réponds de manière structurée, pédagogique et concise et donne des recommandations personnalisées.

Réponse :
"""
        
        raw_response = ai_service.generate_with_ollama(prompt)
        final_response = ai_service.postprocess_response(raw_response)
        
        # Sauvegarder la conversation
        db_instance.save_conversation(session_id, question, final_response, 'analysis', context_count, user_id)
        
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



@chat_bp.route('/conversations/<session_id>', methods=['GET'])
@jwt_required()
def get_conversations(session_id):
    try:
        user_id = get_jwt_identity()
        history = db_instance.get_conversation_history(session_id)
        
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



