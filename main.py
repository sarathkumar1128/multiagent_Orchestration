"""
main.py
────────
Entry point for the multi-agent orchestration CLI.

Flow
─────
  1. Generate unique session_id
  2. Setup logger (console + file at logs/orchestration_{session_id}.log)
  3. Coordinator orchestrates:
       a. Each agent builds its Gemini prompt section
       b. PromptAggregator combines into one prompt
       c. GeminiService generates combined output
       d. Manager splits output → dispatches → PythonAgent / ReactAgent / SQLAgent
       e. Each agent writes files to agents/output/{session_id}/{agent_name}/
  4. Print final summary to console

Usage
──────
  python main.py

Environment
────────────
  GEMINI_API_KEY must be set in .env or as an environment variable.
"""

from __future__ import annotations

import uuid

from logger_config import setup_logger
from coordinator import Coordinator


def main() -> None:
    session_id = uuid.uuid4().hex[:12]

    # ── Setup logging FIRST — both console and file ───────────────────────────
    setup_logger(session_id=session_id)

    import logging
    logger = logging.getLogger(__name__)

    logger.info(
        f"[main] pipeline.started | session={session_id}"
    )

    user_input = (
        "Create a production-ready Task Management Application. "
        "Frontend: React. Middleware/Backend: Python FastAPI. "
        "Database: MySQL."
    )

    logger.info(f"[main] user_request | chars={len(user_input)}")

    coordinator = Coordinator()

    result, agent_results = coordinator.execute(
        user_request=user_input,
        session_id=session_id,
    )

    # ── Final console summary ─────────────────────────────────────────────────
    print(f"\n{'═' * 65}")
    print(f"  ORCHESTRATION COMPLETE")
    print(f"{'═' * 65}")
    print(f"  Session ID     : {session_id}")
    print(f"  Gemini chunks  : {result.chunks_consumed}")
    print(f"  Gemini chars   : {result.total_chars:,}")
    print(f"  Finish reason  : {result.finish_reason.value}")
    print(f"{'─' * 65}")

    for agent_key, r in agent_results.items():
        written = r.get("files_written", [])
        failed  = r.get("files_failed", [])
        print(f"\n  [{r.get('agent', agent_key)}]")
        print(f"    ✅ Written : {len(written)} files")
        if failed:
            print(f"    ❌ Failed  : {len(failed)} files")
        for fp in written:
            print(f"       → {fp}")

    print(f"\n{'─' * 65}")
    print(f"  Output   → agents/output/{session_id}/")
    print(f"  Summary  → agents/output/{session_id}/SUMMARY.md")
    print(f"  Log      → logs/orchestration_{session_id}.log")
    print(f"{'═' * 65}\n")

    logger.info(f"[main] pipeline.finished | session={session_id}")


if __name__ == "__main__":
    main()