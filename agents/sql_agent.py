"""
agents/sql_agent.py
────────────────────
Handles all SQL tasks dispatched by the Manager.

Responsibilities
─────────────────
  - Receives a list of (filepath_hint, code) tuples from the Manager
  - Writes each file under  output/{session_id}/sql_agent/<filepath>
  - Analyses statement types and referenced tables
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

SQL_STATEMENT_KEYWORDS = [
    "SELECT", "INSERT", "UPDATE", "DELETE",
    "CREATE", "DROP", "ALTER", "TRUNCATE",
    "MERGE", "WITH", "INDEX", "USE",
]


class SQLAgent:

    AGENT_NAME = "sql_agent"

    def __init__(self):
        logger.debug(f"[{self.AGENT_NAME}] agent.initialised")

    def build_section(self) -> str:
        """
        Returns the prompt section that instructs Gemini how to structure
        SQL output so the Manager can reliably parse it.
        """
        return """
MySQL Database Requirements:
- Provide a complete schema creation script
- Include all tables, indexes, and foreign keys
- Label the SQL section as: ## 3. MySQL Schema Script
- Use proper MySQL syntax with IF NOT EXISTS guards
- Include migration-safe comments
"""

    def write_code(self, blocks: list[tuple[str, str]], session_id: str) -> dict:
        """
        Write SQL code blocks to disk.

        Parameters
        ----------
        blocks      : list of (filepath_hint, code_content) tuples
        session_id  : used to namespace the output folder

        Returns
        -------
        dict with keys: agent, files_written, files_failed,
                        statements_detected, tables_detected, duration_seconds
        """
        started    = time.monotonic()
        written    = []
        failed     = []
        statements = []
        tables     = []
        agent_dir  = OUTPUT_ROOT / session_id / self.AGENT_NAME

        # ── EVENT: agent task started ────────────────────────────────────────
        logger.info(
            f"[{self.AGENT_NAME}] task.started | "
            f"session={session_id} | blocks={len(blocks)} | "
            f"output_dir={agent_dir}"
        )

        for filepath_hint, code in blocks:
            file_path = agent_dir / filepath_hint
            try:
                # ── Analyse SQL content ───────────────────────────────────────
                upper_code    = code.upper()
                found_stmts   = [
                    kw for kw in SQL_STATEMENT_KEYWORDS
                    if re.search(rf"\b{kw}\b", upper_code)
                ]
                found_tables  = list(set(re.findall(
                    r"(?:FROM|JOIN|INTO|UPDATE|TABLE|USE)\s+([a-zA-Z_][a-zA-Z0-9_]*)",
                    code, re.IGNORECASE,
                )))
                statements.extend(found_stmts)
                tables.extend(found_tables)

                logger.info(
                    f"[{self.AGENT_NAME}] sql.analysed | "
                    f"file={filepath_hint} | "
                    f"statements={found_stmts} | "
                    f"tables={found_tables}"
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
            f"statements={list(set(statements))} | "
            f"tables={list(set(tables))} | "
            f"duration={duration}s"
        )

        return {
            "agent":               self.AGENT_NAME,
            "files_written":       written,
            "files_failed":        failed,
            "statements_detected": list(set(statements)),
            "tables_detected":     list(set(tables)),
            "duration_seconds":    duration,
        }