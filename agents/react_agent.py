# React Agent implementation

class ReactAgent:

    def build_section(self):
        return """
ROLE: Senior React Architect

Use:
- React with Vite
- JavaScript
- Axios

Consume API:
GET    /api/v1/tasks
POST   /api/v1/tasks
PUT    /api/v1/tasks/{id}
DELETE /api/v1/tasks/{id}

Requirements:
- Feature-based folder structure
- TaskList component
- TaskForm component
- TaskItem component
- Loading and error handling
- .env support for API base URL
- Production-ready package.json
- npm install and run instructions
"""
