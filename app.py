import os
import logging
from datetime import timedelta

from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_login import LoginManager
from flask_cors import CORS

# Configure logging
logging.basicConfig(level=logging.DEBUG)

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

# create the app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1) # needed for url_for to generate with https

# CORS configuration for mobile app API access
CORS(app, origins=['*'], supports_credentials=True)

# configure the database
app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

# Session configuration
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)

# Configure upload settings
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['REPORTS_FOLDER'] = 'reports'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create directories if they don't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['REPORTS_FOLDER'], exist_ok=True)

# initialize the app with the extension
db.init_app(app)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
# Type ignore for Railway deployment compatibility
login_manager.login_view = 'health_check'  # type: ignore
login_manager.login_message = 'Authentication required.'
login_manager.login_message_category = 'info'

@login_manager.user_loader
def load_user(user_id):
    try:
        from models import User
        return User.query.get(user_id)
    except ImportError:
        # Return None if models not available
        return None

def safe_import_models():
    """Safely import all models with error handling"""
    try:
        # Try simplified models first for Railway deployment
        import models_simple as models  # noqa: F401
        logging.info("Simplified models imported successfully")
        return True
    except ImportError:
        try:
            # Fallback to full models
            import models  # noqa: F401
            logging.info("Full models imported successfully")
            return True
        except ImportError as e:
            logging.error(f"Failed to import any models: {e}")
            return False
    except Exception as e:
        logging.error(f"Unexpected error importing models: {e}")
        return False

def initialize_database():
    """Initialize database with proper error handling"""
    try:
        with app.app_context():
            # Safe model import
            if safe_import_models():
                # Create all tables
                db.create_all()
                logging.info("Database tables created successfully")
            else:
                logging.warning("Database initialization skipped due to model import failure")
    except Exception as e:
        logging.error(f"Database initialization failed: {e}")

def safe_import_routes():
    """Safely import routes with error handling"""
    try:
        import routes  # noqa: F401
        logging.info("Main routes imported successfully")
    except ImportError as e:
        logging.warning(f"Routes not available: {e}")
    except Exception as e:
        logging.error(f"Unexpected error importing routes: {e}")

# Basic health check route for deployment testing
@app.route('/')
def health_check():
    return {
        "status": "healthy",
        "message": "Optikos Backend API",
        "version": "1.0.0",
        "database": "connected" if db else "unavailable"
    }

@app.route('/health')
def health():
    return {"status": "ok"}

# Initialize everything
initialize_database()
safe_import_routes()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
