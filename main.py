from app import app, db

# Import models and routes separately to control order
with app.app_context():
    # Import models first (no routes dependency)
    import models
    # Import routes after models are registered
    import routes
    # Create tables after everything is imported
    db.create_all()
    print("Database tables created successfully")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
