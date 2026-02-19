"""
manager.py
───────────
Enterprise Manager — reads the orchestration_result.txt produced by the
Coordinator, intelligently splits it into typed code blocks, dispatches
each block to the correct agent, and writes a final summary report.

Parsing Strategy (matches real Gemini output format)
─────────────────────────────────────────────────────
Gemini labels files using markdown headings like:
    #### `backend/app/main.py`
    ```python
    ...code...
    ```

The Manager:
  1. Finds every fenced code block  (``` lang ... ```)
  2. Looks at the heading immediately above to extract the filepath hint
  3. Routes by language tag → python → PythonAgent
                            → jsx/css/js → ReactAgent
                            → sql → SQLAgent
                            → json/text → handled inline (config files)
  4. Dispatches to the agent with (filepath_hint, code, lang) tuples
  5. Writes output/{session_id}/{agent_name}/<filepath> on disk
  6. Logs every single event to both console and log file
  7. Writes a Markdown summary report to output/{session_id}/SUMMARY.md

Output Folder Structure
────────────────────────
output/
└── {session_id}/
    ├── python_agent/
    │   └── backend/
    │       └── app/
    │           ├── main.py
    │           └── ...
    ├── react_agent/
    │   └── frontend/
    │       └── src/
    │           ├── App.jsx
    │           └── ...
    ├── sql_agent/
    │   └── schema.sql
    └── SUMMARY.md
"""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import Optional

from agents.python_agent import PythonAgent
from agents.react_agent import ReactAgent
from agents.sql_agent import SQLAgent

logger = logging.getLogger(__name__)

OUTPUT_ROOT = Path(__file__).parent / "agents" / "output"

# ── Language routing table ────────────────────────────────────────────────────
PYTHON_LANGS  = {"python", "py"}
REACT_LANGS   = {"jsx", "tsx", "react", "js", "javascript", "css"}
SQL_LANGS     = {"sql", "mysql", "pgsql", "sqlite"}
CONFIG_LANGS  = {"json", "text", ""}   # written as config files, not code

# ── Regex: fenced code block with optional language tag ──────────────────────
# Captures: lang (group 1), code body (group 2)
CODE_BLOCK_RE = re.compile(
    r"```([a-zA-Z0-9+#]*)\n(.*?)```",
    re.DOTALL,
)

# ── Regex: heading immediately before a code block ───────────────────────────
# Matches lines like:  #### `backend/app/main.py`
#                      ### `frontend/src/App.jsx`
#                      **`schema.sql`**
FILEPATH_HEADING_RE = re.compile(
    r"(?:#{1,6}\s+|^\*\*)`([^`]+)`",
    re.MULTILINE,
)


class Manager:
    """
    Central orchestrator that parses Gemini output and dispatches to agents.
    """

    def __init__(self):
        self.python_agent = PythonAgent()
        self.react_agent  = ReactAgent()
        self.sql_agent    = SQLAgent()

        logger.info("[manager] manager.initialised")

    # ── Public API ─────────────────────────────────────────────────────────────

    def split_and_write_from_file(self, result_file: str, session_id: str) -> dict:
        """
        Read orchestration result from file, parse and dispatch to agents.

        Parameters
        ----------
        result_file : path to orchestration_result.txt
        session_id  : unique ID for this run (used to namespace output folders)

        Returns
        -------
        dict — full summary of what every agent produced
        """
        # ── EVENT: reading result file ────────────────────────────────────────
        logger.info(
            f"[manager] result_file.reading | "
            f"file={result_file} | session={session_id}"
        )

        if not os.path.exists(result_file):
            logger.critical(
                f"[manager] result_file.not_found | file={result_file}"
            )
            raise FileNotFoundError(
                f"Orchestration result file not found: {result_file}"
            )

        with open(result_file, "r", encoding="utf-8") as f:
            content = f.read()

        logger.info(
            f"[manager] result_file.loaded | "
            f"chars={len(content)} | lines={content.count(chr(10))}"
        )

        return self.split_and_write(content, session_id)

    def split_and_write(self, gemini_output: str, session_id: str) -> dict:
        """
        Parse gemini_output, split by agent type, dispatch, return summary.
        """
        started = time.monotonic()

        # ── EVENT: orchestration started ──────────────────────────────────────
        logger.info(
            f"[manager] orchestration.started | "
            f"session={session_id} | input_chars={len(gemini_output)}"
        )

        # ── Step 1: Parse all code blocks ─────────────────────────────────────
        python_blocks, react_blocks, sql_blocks, config_blocks, skipped = \
            self._parse_blocks(gemini_output)

        logger.info(
            f"[manager] blocks.classified | "
            f"python={len(python_blocks)} | "
            f"react={len(react_blocks)} | "
            f"sql={len(sql_blocks)} | "
            f"config={len(config_blocks)} | "
            f"skipped={skipped}"
        )

        # ── Step 2: Dispatch to agents ─────────────────────────────────────────
        results = {}

        if python_blocks:
            logger.info(
                f"[manager] dispatch.python | "
                f"session={session_id} | blocks={len(python_blocks)}"
            )
            results["python"] = self.python_agent.write_code(
                python_blocks, session_id
            )
        else:
            logger.warning(
                f"[manager] dispatch.python.skipped | "
                f"reason=no_python_blocks_found"
            )

        if react_blocks:
            logger.info(
                f"[manager] dispatch.react | "
                f"session={session_id} | blocks={len(react_blocks)}"
            )
            results["react"] = self.react_agent.write_code(
                react_blocks, session_id
            )
        else:
            logger.warning(
                f"[manager] dispatch.react.skipped | "
                f"reason=no_react_blocks_found"
            )

        if sql_blocks:
            logger.info(
                f"[manager] dispatch.sql | "
                f"session={session_id} | blocks={len(sql_blocks)}"
            )
            results["sql"] = self.sql_agent.write_code(
                sql_blocks, session_id
            )
        else:
            logger.warning(
                f"[manager] dispatch.sql.skipped | "
                f"reason=no_sql_blocks_found"
            )

        # ── Step 3: Write config/misc files directly ──────────────────────────
        if config_blocks:
            config_results = self._write_config_files(config_blocks, session_id)
            results["config"] = config_results

        # ── Step 4: Build and write summary ───────────────────────────────────
        duration = round(time.monotonic() - started, 3)
        summary  = self._build_summary(results, session_id, duration)
        self._write_summary(summary, session_id)

        # ── EVENT: orchestration complete ─────────────────────────────────────
        total_written = sum(
            len(r.get("files_written", [])) for r in results.values()
        )
        total_failed = sum(
            len(r.get("files_failed", [])) for r in results.values()
        )
        logger.info(
            f"[manager] orchestration.complete | "
            f"session={session_id} | "
            f"total_files_written={total_written} | "
            f"total_files_failed={total_failed} | "
            f"duration={duration}s"
        )

        return results

    # ── Parsing ────────────────────────────────────────────────────────────────

    def _parse_blocks(
        self, text: str
    ) -> tuple[list, list, list, list, int]:
        """
        Scan the full Gemini output for fenced code blocks.
        For each block, extract the filepath hint from the heading above it.
        Route to the correct list by language tag.

        Returns
        -------
        python_blocks  : list of (filepath, code)
        react_blocks   : list of (filepath, code, lang)
        sql_blocks     : list of (filepath, code)
        config_blocks  : list of (filepath, code, lang)
        skipped        : count of blocks with unknown/unhandled lang tags
        """
        python_blocks  = []
        react_blocks   = []
        sql_blocks     = []
        config_blocks  = []
        skipped        = 0

        # Counters for auto-naming when no filepath heading found
        counters = {"python": 0, "react": 0, "sql": 0, "config": 0}

        for match in CODE_BLOCK_RE.finditer(text):
            lang = match.group(1).strip().lower()
            code = match.group(2).strip()

            if not code:
                continue

            # Extract filepath hint from the heading just before this block
            filepath_hint = self._extract_filepath_hint(
                text, match.start(), lang, counters
            )

            # ── EVENT: block found ────────────────────────────────────────────
            logger.info(
                f"[manager] block.found | "
                f"lang={lang or '(none)'} | "
                f"filepath={filepath_hint} | "
                f"chars={len(code)}"
            )

            # ── Route by language ─────────────────────────────────────────────
            if lang in PYTHON_LANGS:
                python_blocks.append((filepath_hint, code))
                logger.debug(
                    f"[manager] block.routed | lang={lang} → python_agent"
                )

            elif lang in REACT_LANGS:
                react_blocks.append((filepath_hint, code, lang))
                logger.debug(
                    f"[manager] block.routed | lang={lang} → react_agent"
                )

            elif lang in SQL_LANGS:
                sql_blocks.append((filepath_hint, code))
                logger.debug(
                    f"[manager] block.routed | lang={lang} → sql_agent"
                )

            elif lang in CONFIG_LANGS or not lang:
                config_blocks.append((filepath_hint, code, lang))
                logger.debug(
                    f"[manager] block.routed | lang={lang or 'none'} → config"
                )

            else:
                skipped += 1
                logger.warning(
                    f"[manager] block.unrouted | "
                    f"lang={lang} | filepath={filepath_hint} | "
                    f"reason=no_agent_for_language"
                )

        return python_blocks, react_blocks, sql_blocks, config_blocks, skipped

    def _extract_filepath_hint(
        self,
        text: str,
        block_start: int,
        lang: str,
        counters: dict,
    ) -> str:
        """
        Look backwards from the code block start to find a heading like:
            #### `backend/app/main.py`

        Falls back to an auto-generated name if no heading is found.
        """
        # Search the 400 chars preceding the block for a filepath heading
        preceding = text[max(0, block_start - 400): block_start]
        headings  = FILEPATH_HEADING_RE.findall(preceding)

        if headings:
            raw = headings[-1].strip()
            # Clean up any surrounding backticks
            raw = raw.strip("`")
            # If it looks like a path, use it directly
            if "/" in raw or raw.endswith(
                (".py", ".jsx", ".tsx", ".js", ".css",
                 ".sql", ".json", ".txt", ".md", ".env",
                 ".sh", ".toml", ".yaml", ".yml")
            ):
                return raw

        # ── Fallback: auto-name by language ──────────────────────────────────
        ext_map = {
            "python": "py", "py": "py",
            "jsx": "jsx", "tsx": "tsx", "react": "jsx",
            "js": "js", "javascript": "js",
            "css": "css",
            "sql": "sql", "mysql": "sql",
            "json": "json",
        }
        ext = ext_map.get(lang, "txt")

        if lang in PYTHON_LANGS:
            counters["python"] += 1
            return f"module_{counters['python']}.py"
        elif lang in REACT_LANGS:
            counters["react"] += 1
            return f"Component_{counters['react']}.{ext}"
        elif lang in SQL_LANGS:
            counters["sql"] += 1
            return f"schema_{counters['sql']}.sql"
        else:
            counters["config"] += 1
            return f"config_{counters['config']}.{ext}"

    # ── Config file writer ─────────────────────────────────────────────────────

    def _write_config_files(
        self, blocks: list[tuple[str, str, str]], session_id: str
    ) -> dict:
        """Write JSON / plain text config files to python_agent folder."""
        written = []
        failed  = []
        base    = OUTPUT_ROOT / session_id / "python_agent"

        for filepath_hint, code, lang in blocks:
            file_path = base / filepath_hint
            try:
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(code, encoding="utf-8")
                logger.info(
                    f"[manager] config.written | "
                    f"path={file_path} | lang={lang}"
                )
                written.append(str(file_path))
            except Exception as exc:
                logger.error(
                    f"[manager] config.failed | "
                    f"path={file_path} | error={exc}"
                )
                failed.append(filepath_hint)

        return {"agent": "config", "files_written": written, "files_failed": failed}

    # ── Summary ────────────────────────────────────────────────────────────────

    def _build_summary(
        self, results: dict, session_id: str, duration: float
    ) -> str:
        lines = [
            "# Orchestration Summary",
            "",
            f"**Session ID** : `{session_id}`",
            f"**Duration**   : {duration}s",
            "",
            "## Agent Results",
            "",
        ]

        total_written = 0
        total_failed  = 0

        for agent_key, result in results.items():
            written = result.get("files_written", [])
            failed  = result.get("files_failed",  [])
            total_written += len(written)
            total_failed  += len(failed)

            lines.append(f"### {result.get('agent', agent_key)}")
            lines.append(f"- Files written : **{len(written)}** ✅")
            lines.append(f"- Files failed  : **{len(failed)}** {'❌' if failed else '✅'}")
            if result.get("components_detected"):
                lines.append(
                    f"- Components    : {', '.join(result['components_detected'])}"
                )
            if result.get("statements_detected"):
                lines.append(
                    f"- SQL statements: {', '.join(result['statements_detected'])}"
                )
            if result.get("tables_detected"):
                lines.append(
                    f"- Tables        : {', '.join(result['tables_detected'])}"
                )
            lines.append(f"- Duration      : {result.get('duration_seconds', '—')}s")
            lines.append("")
            if written:
                lines.append("**Files:**")
                for f in written:
                    lines.append(f"  - `{Path(f).name}`  →  `{f}`")
            if failed:
                lines.append("**Failed:**")
                for f in failed:
                    lines.append(f"  - ❌ `{f}`")
            lines.append("")

        lines += [
            "---",
            f"**Total files written** : {total_written}",
            f"**Total files failed**  : {total_failed}",
            "",
            "## Output Structure",
            "```",
            f"agents/output/{session_id}/",
            "├── python_agent/   ← .py files + config",
            "├── react_agent/    ← .jsx / .css / .js files",
            "└── sql_agent/      ← .sql schema files",
            "```",
        ]
        return "\n".join(lines)

    def _write_summary(self, summary: str, session_id: str) -> None:
        summary_dir  = OUTPUT_ROOT / session_id
        summary_dir.mkdir(parents=True, exist_ok=True)
        summary_file = summary_dir / "SUMMARY.md"
        summary_file.write_text(summary, encoding="utf-8")
        logger.info(
            f"[manager] summary.written | file={summary_file}"
        )


# ── Standalone entry point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    import uuid
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from logger_config import setup_logger

    session_id  = uuid.uuid4().hex[:8]
    result_file = "orchestration_result.txt"

    setup_logger(session_id=session_id)

    manager = Manager()
    manager.split_and_write_from_file(result_file, session_id)

    print(f"\n✅ Manager processed session: {session_id}")
    print(f"   Output  → agents/output/{session_id}/")
    print(f"   Summary → agents/output/{session_id}/SUMMARY.md")
    print(f"   Log     → logs/orchestration_{session_id}.log")