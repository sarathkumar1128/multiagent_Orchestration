"""
prompt_aggregator.py
─────────────────────
Combines the user request with each agent's prompt section into the
final Gemini prompt. Instructs Gemini to label every file so the
Manager can reliably parse and route the output.
"""

from __future__ import annotations


class PromptAggregator:

    @staticmethod
    def combine(
        user_request: str,
        sql_section: str,
        backend_section: str,
        frontend_section: str,
    ) -> str:
        return f"""
You are a Principal Full-Stack Engineer.

USER REQUEST:
{user_request}

==================================================
DATABASE SECTION
==================================================
{sql_section}

==================================================
BACKEND SECTION
==================================================
{backend_section}

==================================================
FRONTEND SECTION
==================================================
{frontend_section}

==================================================
OUTPUT FORMAT RULES — READ CAREFULLY
==================================================
You MUST label every single file using this exact format before each code block:

#### `path/to/filename.ext`
```language
...code...
```

Examples:
#### `backend/app/main.py`
```python
from fastapi import FastAPI
app = FastAPI()
```

#### `frontend/src/App.jsx`
```jsx
import React from 'react';
export default function App() {{ return <div>Hello</div>; }}
```

#### `schema.sql`
```sql
CREATE TABLE tasks (...);
```

==================================================
DELIVERABLES REQUIRED
==================================================
1. Full backend implementation (FastAPI + SQLAlchemy)
2. Full frontend implementation (React + Vite)
3. MySQL schema script
4. requirements.txt
5. package.json
6. .env.example files (backend + frontend)
7. Run instructions

Rules:
- No placeholders. No pseudo-code. Complete working code only.
- Every file must have the #### `filepath` heading immediately above its code block.
- No explanations outside code blocks except for the run instructions section.
"""