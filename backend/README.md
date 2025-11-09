# Backend Installation and Setup

## Install Dependencies
```bash
cd backend
pip install -r requirements.txt
```

## Run Server
```bash
python server.py
```

Backend API will run on: http://localhost:5000

## API Endpoints

### POST /api/chat
Send chat message to AI assistant

Request:
```json
{
  "message": "Assess damage from Hurricane Ian in Fort Myers"
}
```

Response:
```json
{
  "response": "AI response text...",
  "assessment": {
    "damage_level": "Major Damage",
    "confidence": "85.3%",
    "decision": "APPROVE"
  }
}
```

### POST /api/analyze-image
Upload image for damage analysis

Request: multipart/form-data with 'image' field

Response: Same as /api/chat

### POST /api/sample-analysis
Run sample Hurricane Ian analysis

Response: Same as /api/chat

### GET /api/health
Health check

Response:
```json
{
  "status": "healthy"
}
```
