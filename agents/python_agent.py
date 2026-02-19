"""
agents/python_agent.py
───────────────────────
Handles all Python code tasks dispatched by the Manager.

Responsibilities
─────────────────
  - Receives a list of (filepath_hint, code) tuples from the Manager
  - Writes each file under  output/{session_id}/python_agent/<filepath>
  - Validates Python syntax before writing
  - Logs every event at the correct level
  - Returns a summary dict for the Manager's final report
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)

OUTPUT_ROOT = Path(__file__).parent / "output"


class PythonAgent:

    AGENT_NAME = "python_agent"

    def __init__(self):
        logger.debug(f"[{self.AGENT_NAME}] agent.initialised")

    def build_section(self) -> str:
        """
        Returns the prompt section that instructs Gemini how to structure
        Python output so the Manager can reliably parse it.
        """
        return """
Python Backend Requirements:
- Use FastAPI framework
- Include SQLAlchemy ORM models
- Provide full CRUD operations
- Label each file with its exact path as: #### `backend/path/to/file.py`
- Every file must be a complete, runnable implementation — no placeholders
"""

    def write_code(self, blocks: list[tuple[str, str]], session_id: str) -> dict:
        """
        Write Python code blocks to disk.

        Parameters
        ----------
        blocks      : list of (filepath_hint, code_content) tuples
        session_id  : used to namespace the output folder

        Returns
        -------
        dict with keys: agent, files_written, files_failed, duration_seconds
        """
        started   = time.monotonic()
        written   = []
        failed    = []
        agent_dir = OUTPUT_ROOT / session_id / self.AGENT_NAME

        # ── EVENT: agent task started ────────────────────────────────────────
        logger.info(
            f"[{self.AGENT_NAME}] task.started | "
            f"session={session_id} | blocks={len(blocks)} | "
            f"output_dir={agent_dir}"
        )

        for filepath_hint, code in blocks:
            file_path = agent_dir / filepath_hint
            try:
                # ── Validate Python syntax ────────────────────────────────────
                try:
                    compile(code, str(filepath_hint), "exec")
                    logger.info(
                        f"[{self.AGENT_NAME}] syntax.valid | file={filepath_hint}"
                    )
                except SyntaxError as syn_err:
                    logger.warning(
                        f"[{self.AGENT_NAME}] syntax.warning | "
                        f"file={filepath_hint} | error={syn_err}"
                    )

                # ── Write file ────────────────────────────────────────────────
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(code, encoding="utf-8")

                logger.info(
                    f"[{self.AGENT_NAME}] file.written | "
                    f"path={file_path} | chars={len(code)}"
                )
                written.append(str(file_path))

            except Exception as exc:
                logger.error(
                    f"[{self.AGENT_NAME}] file.failed | "
                    f"path={file_path} | error={exc}"
                )
                failed.append(str(filepath_hint))

        duration = round(time.monotonic() - started, 3)

        # ── EVENT: agent task complete ────────────────────────────────────────
        logger.info(
            f"[{self.AGENT_NAME}] task.complete | "
            f"session={session_id} | "
            f"written={len(written)} | failed={len(failed)} | "
            f"duration={duration}s"
        )

        return {
            "agent":            self.AGENT_NAME,
            "files_written":    written,
            "files_failed":     failed,
            "duration_seconds": duration,
        }