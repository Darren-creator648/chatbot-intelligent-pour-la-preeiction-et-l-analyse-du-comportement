from flask import Flask
from flask_cors import CORS
from flask_jwt_extended import JWTManager
import threading
from config import Config

def create_app():
    """Factory pour créer l'application Flask"""
    app = Flask(__name__)
    CORS(app)
    
    # Configuration
    app.config['JWT_SECRET_KEY'] = Config.JWT_SECRET_KEY
    app.config['JWT_ACCESS_TOKEN_EXPIRES'] = Config.JWT_ACCESS_TOKEN_EXPIRES
    
    # Initialisation JWT
    jwt = JWTManager(app)
    
    # Enregistrement des blueprints
    from routes.auth import auth_bp
    from routes.chat import chat_bp
    from routes.admin import admin_bp
    from routes.stats import stats_bp
    
    app.register_blueprint(stats_bp, url_prefix='/api')

    app.register_blueprint(auth_bp, url_prefix='/api')
    app.register_blueprint(chat_bp, url_prefix='/api')
    app.register_blueprint(admin_bp, url_prefix='/api/admin')
    
    return app

def init_background():
    """Initialise le système IA en arrière-plan"""
    
    from services.ai_service import ai_service
    ai_service.initialize_system()
    

if __name__ == '__main__':
    # Créer l'application
    app = create_app()
    
    # Lancer l'initialisation en arrière-plan
    init_thread = threading.Thread(target=init_background)
    init_thread.daemon = True
    init_thread.start()
    
    print("🚀 Démarrage du serveur Flask...")
    print("🔧 Initialisation en cours en arrière-plan...")
    print("🔐 Authentification JWT activée")
    print("🗄️ Base de données MongoDB configurée")
    
    app.run(
        debug=Config.DEBUG, 
        host=Config.HOST, 
        port=Config.PORT
    )