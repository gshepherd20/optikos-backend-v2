from datetime import datetime, timedelta
from app import db
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
import uuid
import jwt
import os

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    first_name = db.Column(db.String(50))
    last_name = db.Column(db.String(50))
    company = db.Column(db.String(100))
    phone = db.Column(db.String(20))
    
    # Subscription fields
    subscription_tier = db.Column(db.String(20), default='free')  # free, basic, pro
    subscription_status = db.Column(db.String(20), default='active')  # active, cancelled, expired
    subscription_start = db.Column(db.DateTime)
    subscription_end = db.Column(db.DateTime)
    stripe_customer_id = db.Column(db.String(100))
    stripe_subscription_id = db.Column(db.String(100))
    
    # Usage tracking
    analyses_used = db.Column(db.Integer, default=0)
    analyses_limit = db.Column(db.Integer, default=3)  # Free tier limit
    
    # Account status
    is_verified = db.Column(db.Boolean, default=False)
    is_active = db.Column(db.Boolean, default=True)
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_login = db.Column(db.DateTime)
    
    # Relationships
    analyses = db.relationship('Analysis', backref='user', lazy=True, cascade='all, delete-orphan')
    api_keys = db.relationship('APIKey', backref='user', lazy=True, cascade='all, delete-orphan')

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
    
    def get_full_name(self):
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        return self.email
    
    def can_analyze(self):
        """Check if user can perform analysis based on subscription"""
        if self.subscription_tier in ['basic', 'pro']:
            return True
        # Free tier check
        return self.analyses_used < self.analyses_limit
    
    def get_analyses_remaining(self):
        """Get remaining analyses for current period"""
        if self.subscription_tier in ['basic', 'pro']:
            return 999  # Unlimited for paid tiers
        return max(0, self.analyses_limit - self.analyses_used)
    
    def is_subscription_active(self):
        """Check if subscription is currently active"""
        if self.subscription_tier == 'free':
            return True
        return (self.subscription_status == 'active' and 
                self.subscription_end and 
                self.subscription_end > datetime.utcnow())
    
    def generate_api_token(self, expires_in=3600):
        """Generate JWT token for API access"""
        payload = {
            'user_id': self.id,
            'exp': datetime.utcnow() + timedelta(seconds=expires_in),
            'iat': datetime.utcnow()
        }
        return jwt.encode(payload, os.environ.get('SESSION_SECRET', 'dev-secret'), algorithm='HS256')
    
    @staticmethod
    def verify_api_token(token):
        """Verify JWT token and return user"""
        try:
            payload = jwt.decode(token, os.environ.get('SESSION_SECRET', 'dev-secret'), algorithms=['HS256'])
            return User.query.get(payload['user_id'])
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

    def to_dict(self):
        """Convert user to dictionary for API responses"""
        return {
            'id': self.id,
            'email': self.email,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'company': self.company,
            'subscription_tier': self.subscription_tier,
            'subscription_status': self.subscription_status,
            'analyses_used': self.analyses_used,
            'analyses_remaining': self.get_analyses_remaining(),
            'is_verified': self.is_verified,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class APIKey(db.Model):
    __tablename__ = 'api_keys'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    key_hash = db.Column(db.String(256), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    is_active = db.Column(db.Boolean, default=True)
    last_used = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    expires_at = db.Column(db.DateTime)

    def set_key(self, key):
        self.key_hash = generate_password_hash(key)
    
    def check_key(self, key):
        return check_password_hash(self.key_hash, key)

class Analysis(db.Model):
    __tablename__ = 'analyses'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    
    # Analysis metadata
    name = db.Column(db.String(200))
    description = db.Column(db.Text)
    
    # Image information
    original_image_path = db.Column(db.String(500))
    replacement_image_path = db.Column(db.String(500))
    
    # Analysis results
    points_analyzed = db.Column(db.Integer, default=0)
    average_delta_e = db.Column(db.Float)
    average_texture_delta = db.Column(db.Float)
    average_gloss_delta = db.Column(db.Float)
    perceptos_index = db.Column(db.Float)
    uniformity_assessment = db.Column(db.String(20))  # uniform, non-uniform
    
    # Analysis data (JSON)
    analysis_data = db.Column(db.JSON)
    
    # Status and timestamps
    status = db.Column(db.String(20), default='pending')  # pending, processing, completed, failed
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime)
    
    def to_dict(self):
        """Convert analysis to dictionary for API responses"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'points_analyzed': self.points_analyzed,
            'average_delta_e': self.average_delta_e,
            'average_texture_delta': self.average_texture_delta,
            'average_gloss_delta': self.average_gloss_delta,
            'perceptos_index': self.perceptos_index,
            'uniformity_assessment': self.uniformity_assessment,
            'status': self.status,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'analysis_data': self.analysis_data
        }

class Subscription(db.Model):
    __tablename__ = 'subscriptions'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    
    # Stripe information
    stripe_subscription_id = db.Column(db.String(100), unique=True)
    stripe_customer_id = db.Column(db.String(100))
    stripe_price_id = db.Column(db.String(100))
    
    # Subscription details
    tier = db.Column(db.String(20), nullable=False)  # basic, pro
    status = db.Column(db.String(20), nullable=False)  # active, cancelled, past_due, unpaid
    current_period_start = db.Column(db.DateTime)
    current_period_end = db.Column(db.DateTime)
    
    # Billing
    amount = db.Column(db.Integer)  # Amount in cents
    currency = db.Column(db.String(3), default='USD')
    interval = db.Column(db.String(20))  # month, year
    
    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    cancelled_at = db.Column(db.DateTime)
    
    user = db.relationship('User', backref='subscription_records')

class EmailVerification(db.Model):
    __tablename__ = 'email_verifications'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    token = db.Column(db.String(100), nullable=False, unique=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    expires_at = db.Column(db.DateTime, nullable=False)
    used_at = db.Column(db.DateTime)
    
    user = db.relationship('User', backref='verification_tokens')
    
    def is_expired(self):
        return datetime.utcnow() > self.expires_at
    
    def is_used(self):
        return self.used_at is not None

class PasswordReset(db.Model):
    __tablename__ = 'password_resets'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(db.String(36), db.ForeignKey('users.id'), nullable=False)
    token = db.Column(db.String(100), nullable=False, unique=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    expires_at = db.Column(db.DateTime, nullable=False)
    used_at = db.Column(db.DateTime)
    
    user = db.relationship('User', backref='reset_tokens')
    
    def is_expired(self):
        return datetime.utcnow() > self.expires_at
    
    def is_used(self):
        return self.used_at is not None