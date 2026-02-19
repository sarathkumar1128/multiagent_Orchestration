# SQL Agent implementation

class SQLAgent:

    def build_section(self):
        return """
ROLE: Database Architect

Generate MySQL schema.

Database:
CREATE DATABASE task_management;

Table: tasks
- id INT AUTO_INCREMENT PRIMARY KEY
- title VARCHAR(255) NOT NULL
- description TEXT
- status ENUM('pending','completed') DEFAULT 'pending'
- created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
- updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP

Add index on status.
Include migration-safe notes.
"""
