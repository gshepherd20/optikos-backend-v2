from app import app, db

# Import models BEFORE importing routes to prevent circular imports
import models

# Import routes after models are loaded
import routes

# Create tables only once at startup
with app.app_context():
    db.create_all()
    print("Database tables created successfully")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
