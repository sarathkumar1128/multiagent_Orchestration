# Python Agent implementation
class PythonAgent:

    def build_section(self):
        return """
ROLE: Senior Python Backend Architect

Use FastAPI.

API CONTRACT:
Base path: /api/v1

Endpoints:
GET    /api/v1/tasks
POST   /api/v1/tasks
PUT    /api/v1/tasks/{id}
DELETE /api/v1/tasks/{id}

Response format:
{
  "success": boolean,
  "data": object,
  "message": string
}

Requirements:
- Modular structure
- Service layer
- MySQL connection using environment variables
- CORS enabled
- Logging configuration
- Exception middleware
- requirements.txt included
- Run instructions included
"""

