# Optikos Backend API

Material uniformity analysis backend with Flask and computer vision.

## Features
- Image analysis using OpenCV and scikit-image
- Color difference calculations (Delta-E CIE2000)
- Texture analysis with Local Binary Patterns
- Mobile app API endpoints
- PostgreSQL database storage

## Deployment
Deployed on Railway with PostgreSQL database.

## Environment Variables
- SESSION_SECRET: Flask session encryption key
- DATABASE_URL: PostgreSQL connection (auto-set by Railway)

## API Endpoints
- `/mobile/upload` - Upload images from mobile app
- `/mobile/analyze` - Perform scientific analysis
- `/` - Web interface for desktop use