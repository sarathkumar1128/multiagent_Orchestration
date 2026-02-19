"""
coordinator.py
───────────────
Manages the full agent workflow:
  1. Each agent provides its prompt section (what it needs from Gemini)
  2. PromptAggregator combines them into a single structured prompt
  3. GeminiService generates the full combined output
  4. Result is saved to orchestration_result_{session_id}.txt
  5. Manager reads the result, splits it, and dispatches to agents

Every step is logged.
"""

from __future__ import annotations

import logging
import time

from agents.sql_agent import SQLAgent
from agents.python_agent import PythonAgent
from agents.react_agent import ReactAgent
from prompt_aggregator import PromptAggregator
from gemini_service import GeminiService
from manager import Manager

logger = logging.getLogger(__name__)


class Coordinator:

    def __init__(self):
        self.sql_agent    = SQLAgent()
        self.python_agent = PythonAgent()
        self.react_agent  = ReactAgent()
        self.gemini       = GeminiService()
        self.manager      = Manager()

        logger.info("[coordinator] coordinator.initialised")

    def execute(self, user_request: str, session_id: str):
        started = time.monotonic()

        # ── EVENT: orchestration started ──────────────────────────────────────
        logger.info(
            f"[coordinator] orchestration.started | "
            f"session={session_id} | "
            f"request_chars={len(user_request)}"
        )

        # ── Step 1: Collect prompt sections from each agent ───────────────────
        logger.info("[coordinator] agents.building_sections")

        frontend_section = self.react_agent.build_section()
        logger.info("[coordinator] react_agent.section_ready")

        sql_section = self.sql_agent.build_section()
        logger.info("[coordinator] sql_agent.section_ready")

        backend_section = self.python_agent.build_section()
        logger.info("[coordinator] python_agent.section_ready")

        # ── Step 2: Combine into final prompt ─────────────────────────────────
        final_prompt = PromptAggregator.combine(
            user_request,
            sql_section,
            backend_section,
            frontend_section,
        )
        logger.info(
            f"[coordinator] prompt.combined | chars={len(final_prompt)}"
        )

        # ── Step 3: Save the final prompt for auditability ────────────────────
        prompt_filename = f"finalprompt_{session_id}.txt"
        with open(prompt_filename, "w", encoding="utf-8") as f:
            f.write(final_prompt)
        logger.info(
            f"[coordinator] prompt.saved | file={prompt_filename}"
        )

        # ── Step 4: Call Gemini ───────────────────────────────────────────────
        logger.info("[coordinator] gemini.calling")
        result = self.gemini.generate_until_complete(final_prompt)

        logger.info(
            f"[coordinator] gemini.complete | "
            f"chunks={result.chunks_consumed} | "
            f"chars={result.total_chars} | "
            f"finish_reason={result.finish_reason.value} | "
            f"duration={result.total_duration_seconds}s"
        )

        # ── Step 5: Save Gemini result ────────────────────────────────────────
        result_filename = f"orchestration_result_{session_id}.txt"
        with open(result_filename, "w", encoding="utf-8") as f:
            f.write(result.text)
        logger.info(
            f"[coordinator] result.saved | file={result_filename}"
        )

        # ── Step 6: Hand off to Manager for splitting + agent dispatch ─────────
        logger.info(
            f"[coordinator] manager.dispatching | session={session_id}"
        )
        agent_results = self.manager.split_and_write(result.text, session_id)

        duration = round(time.monotonic() - started, 3)

        # ── EVENT: orchestration complete ─────────────────────────────────────
        total_written = sum(
            len(r.get("files_written", [])) for r in agent_results.values()
        )
        logger.info(
            f"[coordinator] orchestration.complete | "
            f"session={session_id} | "
            f"total_files_written={total_written} | "
            f"total_duration={duration}s"
        )

        return result, agent_results