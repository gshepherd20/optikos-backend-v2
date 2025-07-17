from app import app, db, safe_import_models

# WSGI-safe initialization - only run once per process
def initialize_once():
    if not hasattr(initialize_once, 'done'):
        with app.app_context():
            # Use controlled import to prevent double registration
            if safe_import_models():
                db.create_all()
                print("Database tables created successfully")
            else:
                print("Models already imported, skipping initialization")
        initialize_once.done = True

# Initialize for both WSGI and direct execution
initialize_once()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
