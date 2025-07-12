import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase
from werkzeug.middleware.proxy_fix import ProxyFix
import logging

logging.basicConfig(level=logging.DEBUG)

class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET") or "fallback-dev-key-for-testing"
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///optikos.db")
app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {
    "pool_recycle": 300,
    "pool_pre_ping": True,
}

# Create upload and reports directories
upload_folder = 'uploads'
reports_folder = 'reports'
os.makedirs(upload_folder, exist_ok=True)
os.makedirs(reports_folder, exist_ok=True)

app.config['UPLOAD_FOLDER'] = upload_folder
app.config['REPORTS_FOLDER'] = reports_folder
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize database
db.init_app(app)

# Lazy initialization function to prevent circular imports
def initialize_app():
    with app.app_context():
        import models  # noqa: F401
        import routes  # noqa: F401
        db.create_all()
        print("Database tables created successfully")

# Initialization will be called explicitly from main.py
