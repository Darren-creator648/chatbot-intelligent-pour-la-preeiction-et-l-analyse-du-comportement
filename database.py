from pymongo import MongoClient
from bson import ObjectId
from datetime import datetime
import uuid
from config import Config

class Database:
    def __init__(self):
        self.client = None
        self.db = None
        self.users_collection = None
        self.conversations_collection = None
        self.sessions_collection = None
        self.connect()
    
    def connect(self):
        """Établit la connexion à MongoDB"""
        try:
            self.client = MongoClient(Config.MONGODB_URI)
            self.db = self.client[Config.DATABASE_NAME]
            
            # Collections
            self.users_collection = self.db.users
            self.conversations_collection = self.db.conversations
            self.sessions_collection = self.db.sessions
            
            print("✅ Connexion MongoDB établie")
        except Exception as e:
            print(f"❌ Erreur connexion MongoDB: {e}")
            self.db = None
    
    def serialize_mongo_document(self, doc):
        """Convertit les ObjectId MongoDB en string pour JSON"""
        if doc is None:
            return None
        if isinstance(doc, list):
            return [self.serialize_mongo_document(item) for item in doc]
        if isinstance(doc, dict):
            for key, value in doc.items():
                if isinstance(value, ObjectId):
                    doc[key] = str(value)
                elif isinstance(value, datetime):
                    doc[key] = value.isoformat()
                elif isinstance(value, dict):
                    doc[key] = self.serialize_mongo_document(value)
                elif isinstance(value, list):
                    doc[key] = self.serialize_mongo_document(value)
        return doc
    
    def save_conversation(self, session_id, user_message, bot_response, message_type='analysis', context_used=0, user_id=None):
        """Sauvegarde une conversation"""
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
            
            self.conversations_collection.insert_one(conversation_data)
            
            # Mettre à jour ou créer la session
            self.sessions_collection.update_one(
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
    
    def get_conversation_history(self, session_id, limit=50):
        """Récupère l'historique des conversations"""
        try:
            conversations = self.conversations_collection.find(
                {'session_id': session_id}
            ).sort('timestamp', 1).limit(limit)
            
            return list(conversations)
        except Exception as e:
            print(f"Erreur récupération historique: {e}")
            return []
    
    def get_analytics_data(self, user_id=None):
        """Récupère les données analytiques"""
        try:
            from datetime import timedelta
            
            # Filtrer par utilisateur si spécifié
            filter_query = {'user_id': user_id} if user_id else {}
            
            # Statistiques générales
            total_conversations = self.conversations_collection.count_documents(filter_query)
            total_sessions = self.sessions_collection.count_documents(filter_query)
            
            # Messages par type
            pipeline = [
                {'$match': filter_query},
                {'$group': {'_id': '$message_type', 'count': {'$sum': 1}}}
            ]
            message_types_cursor = self.conversations_collection.aggregate(pipeline)
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
            daily_activity_cursor = self.conversations_collection.aggregate(pipeline)
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

# Instance globale
db_instance = Database()