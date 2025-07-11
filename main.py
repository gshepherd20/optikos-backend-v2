from app import app, db
import routes

# Import models and create tables only once at startup
with app.app_context():
    import models
    db.create_all()
    print("Database tables created successfully")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
