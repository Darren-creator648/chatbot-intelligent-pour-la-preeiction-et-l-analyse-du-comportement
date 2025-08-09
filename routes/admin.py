from flask import Blueprint, jsonify
from flask_jwt_extended import jwt_required, get_jwt
from functools import wraps
from database import db_instance

admin_bp = Blueprint('admin', __name__)

def admin_required(f):
    """Décorateur pour vérifier les droits administrateur"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        claims = get_jwt()
        if claims.get('role') != 'admin':
            return jsonify({'error': 'Accès administrateur requis'}), 403
        return f(*args, **kwargs)
    return decorated_function

@admin_bp.route('/users', methods=['GET'])
@jwt_required()
@admin_required
def get_all_users():
    """Récupère tous les utilisateurs (admin seulement)"""
    try:
        users = db_instance.users_collection.find({}, {'password': 0})  # Exclure les mots de passe
        users_list = [db_instance.serialize_mongo_document(user) for user in users]
        return jsonify({'users': users_list})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@admin_bp.route('/analytics/global', methods=['GET'])
@jwt_required()
@admin_required
def get_global_analytics():
    """Récupère les analytics globales (tous les utilisateurs)"""
    try:
        analytics = db_instance.get_analytics_data()  # Sans filtrage par utilisateur
        return jsonify(analytics)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@admin_bp.route('/system/status', methods=['GET'])
@jwt_required()
@admin_required
def get_system_status():
    """Récupère le statut du système"""
    try:
        from services.ai_service import ai_service
        
        status = {
            'database_connected': db_instance.db is not None,
            'ai_model_loaded': ai_service.model is not None,
            'faiss_index_loaded': ai_service.index is not None,
            'students_data_loaded': len(ai_service.raw_students_data) > 0,
            'chunks_count': len(ai_service.chunks),
            'students_count': len(ai_service.raw_students_data),
            'system_initialized': ai_service.is_initialized()
        }
        
        return jsonify({'system_status': status})
    except Exception as e:
        return jsonify({"error": str(e)}), 500