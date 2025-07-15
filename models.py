from datetime import datetime
from app import db
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
import uuid

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    first_name = db.Column(db.String(50))
    last_name = db.Column(db.String(50))
    
    # Basic subscription info
    subscription_tier = db.Column(db.String(20), default='free')
    analyses_used = db.Column(db.Integer, default=0)
    analyses_limit = db.Column(db.Integer, default=3)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def get_full_name(self):
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        return self.email
    
    def can_analyze(self):
        """Check if user can perform analysis"""
        if self.subscription_tier in ['basic', 'pro']:
            return True
        return self.analyses_used < self.analyses_limit

class Analysis(db.Model):
    __tablename__ = 'analyses'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    
    # Analysis results
    points_analyzed = db.Column(db.Integer, default=0)
    average_delta_e = db.Column(db.Float)
    perceptos_index = db.Column(db.Float)
    uniformity_assessment = db.Column(db.String(20))
    
    # Analysis data (JSON)
    analysis_data = db.Column(db.JSON)
    
    # Status and timestamps
    status = db.Column(db.String(20), default='completed')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationship
    user = db.relationship('User', backref='analyses')
