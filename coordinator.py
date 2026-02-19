# Coordinator for managing agent workflows

import logging
from agents.sql_agent import SQLAgent
from agents.python_agent import PythonAgent
from agents.react_agent import ReactAgent
from prompt_aggregator import PromptAggregator
from gemini_service import GeminiService

class Coordinator:

    def __init__(self):
        self.sql_agent = SQLAgent()
        self.python_agent = PythonAgent()
        self.react_agent = ReactAgent()
        self.gemini = GeminiService()

    def execute(self, user_request: str):

        logging.info("Orchestration started")

        logging.info("Selecting agents")
        sql_section = self.sql_agent.build_section()
        logging.info("SQL Agent executed")

        backend_section = self.python_agent.build_section()
        logging.info("Python Agent executed")

        frontend_section = self.react_agent.build_section()
        logging.info("React Agent executed")

        final_prompt = PromptAggregator.combine(
            user_request,
            sql_section,
            backend_section,
            frontend_section
        )

        logging.info("Final prompt constructed")

        result = self.gemini.generate(final_prompt)

        logging.info("Orchestration completed")

        return result
