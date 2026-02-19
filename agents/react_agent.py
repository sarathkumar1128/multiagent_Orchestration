"""
agents/react_agent.py
──────────────────────
Handles all React / JSX / CSS / JS tasks dispatched by the Manager.

Responsibilities
─────────────────
  - Receives a list of (filepath_hint, code, lang) tuples from the Manager
  - Writes each file under  output/{session_id}/react_agent/<filepath>
  - Detects React component names for the manifest
  - Logs every event at the correct level
  - Returns a summary dict for the Manager's final report
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

logger = logging.getLogger(__name__)

OUTPUT_ROOT = Path(__file__).parent / "output"


class ReactAgent:

    AGENT_NAME = "react_agent"

    # Extensions the React agent owns
    OWNED_LANGS = {"jsx", "tsx", "react", "js", "javascript", "css"}

    def __init__(self):
        logger.debug(f"[{self.AGENT_NAME}] agent.initialised")

    def build_section(self) -> str:
        """
        Returns the prompt section that instructs Gemini how to structure
        React output so the Manager can reliably parse it.
        """
        return """
React Frontend Requirements:
- Use React 18 with functional components and hooks
- Use Vite as the build tool
- Label each file with its exact path as: #### `frontend/src/path/to/Component.jsx`
- Include CSS files labelled as: #### `frontend/src/App.css`
- Every component must be complete and production-ready — no placeholders
"""

    def write_code(self, blocks: list[tuple[str, str, str]], session_id: str) -> dict:
        """
        Write React/JSX/CSS/JS blocks to disk.

        Parameters
        ----------
        blocks      : list of (filepath_hint, code_content, lang) tuples
        session_id  : used to namespace the output folder

        Returns
        -------
        dict with keys: agent, files_written, files_failed,
                        components_detected, duration_seconds
        """
        started    = time.monotonic()
        written    = []
        failed     = []
        components = []
        agent_dir  = OUTPUT_ROOT / session_id / self.AGENT_NAME

        # ── EVENT: agent task started ────────────────────────────────────────
        logger.info(
            f"[{self.AGENT_NAME}] task.started | "
            f"session={session_id} | blocks={len(blocks)} | "
            f"output_dir={agent_dir}"
        )

        for filepath_hint, code, lang in blocks:
            file_path = agent_dir / filepath_hint
            try:
                # ── Detect React component names ──────────────────────────────
                if lang in {"jsx", "tsx", "react"}:
                    found = re.findall(
                        r"(?:function|class|const)\s+([A-Z][a-zA-Z0-9]+)", code
                    )
                    if found:
                        components.extend(found)
                        logger.info(
                            f"[{self.AGENT_NAME}] components.detected | "
                            f"file={filepath_hint} | components={found}"
                        )

                # ── Write file ────────────────────────────────────────────────
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(code, encoding="utf-8")

                logger.info(
                    f"[{self.AGENT_NAME}] file.written | "
                    f"path={file_path} | lang={lang} | chars={len(code)}"
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
            f"components={list(set(components))} | "
            f"duration={duration}s"
        )

        return {
            "agent":               self.AGENT_NAME,
            "files_written":       written,
            "files_failed":        failed,
            "components_detected": list(set(components)),
            "duration_seconds":    duration,
        }