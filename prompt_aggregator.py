# Aggregates prompts from different agents

class PromptAggregator:

    @staticmethod
    def combine(user_request, sql_section, backend_section, frontend_section):
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
OUTPUT REQUIREMENTS
==================================================
Generate:

1. Full backend folder structure
2. Full frontend folder structure
3. MySQL schema script
4. requirements.txt
5. package.json
6. .env.example files
7. Run instructions

No placeholders.
No pseudo-code.
No explanations outside code.
"""
