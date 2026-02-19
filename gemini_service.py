

import os
import logging
import google.genai as genai  # Updated import
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

class GeminiService:

    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise Exception("GEMINI_API_KEY not set")

        self.model_name = "models/gemini-pro"
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(self.model_name)

    def generate(self, prompt: str):
        try:
            logger.info("Gemini SDK call started")
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": 4000,
                    "temperature": 0.5,
                    "top_p": 0.9,
                }
            )
            logger.info("Gemini response received")
            if hasattr(response, 'text'):
                return response.text
            elif hasattr(response, 'result'):
                return response.result
            else:
                return str(response)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str:
                logger.error("Quota exceeded (429)")
                raise Exception("Quota exceeded. Please retry later.")
            if "timeout" in error_str.lower():
                logger.error("Timeout occurred")
                raise Exception("Gemini request timed out.")
            logger.error(f"Gemini API error: {error_str}")
            raise