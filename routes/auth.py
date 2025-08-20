from flask import Blueprint, request, jsonify
from flask_jwt_extended import jwt_required, create_access_token, get_jwt_identity
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
from bson import ObjectId
from database import db_instance

class AuthController:
    """Contrôleur pour la gestion de l'authentification"""
    
    def __init__(self, db_instance):
        self.db = db_instance
    
    def register(self):
        """Enregistrement d'un nouvel utilisateur"""
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
            if self.db.users_collection.find_one({'email': email}):
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
            
            result = self.db.users_collection.insert_one(user_data)
            
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

    def login(self):
        """Connexion d'un utilisateur"""
        try:
            data = request.get_json()
            email = data.get('email', '').strip().lower()
            password = data.get('password', '')
            
            if not email or not password:
                return jsonify({'error': 'Email et mot de passe requis'}), 400
            
            # Trouver l'utilisateur
            user = self.db.users_collection.find_one({'email': email})
            if not user or not check_password_hash(user['password'], password):
                return jsonify({'error': 'Email ou mot de passe incorrect'}), 401
            
            if not user.get('is_active', True):
                return jsonify({'error': 'Compte désactivé'}), 401
            
            # Mettre à jour la dernière connexion
            self.db.users_collection.update_one(
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

    def get_profile(self):
        """Récupération du profil utilisateur"""
        try:
            user_id = get_jwt_identity()
            user = self.db.users_collection.find_one({'_id': ObjectId(user_id)})
            
            if not user:
                return jsonify({'error': 'Utilisateur non trouvé'}), 404
            
            user_data = self.db.serialize_mongo_document(user)
            # Supprimer le mot de passe
            user_data.pop('password', None)
            
            return jsonify({'user': user_data})
            
        except Exception as e:
            return jsonify({'error': f'Erreur serveur: {str(e)}'}), 500

# Créer une instance du contrôleur
auth_controller = AuthController(db_instance)

# Créer le blueprint
auth_bp = Blueprint('auth', __name__)

# Enregistrer les routes avec les méthodes de la classe
auth_bp.route('/register', methods=['POST'])(auth_controller.register)
auth_bp.route('/login', methods=['POST'])(auth_controller.login)
auth_bp.route('/profile', methods=['GET'])(jwt_required()(auth_controller.get_profile))