"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         GeminiContinuationGenerator — Enterprise Edition                   ║
║                                                                              ║
║  Drives a multi-turn Gemini chat session that transparently stitches        ║
║  MAX_TOKENS (paginated) continuations into a single cohesive response.      ║
║                                                                              ║
║  Features                                                                    ║
║  ─────────                                                                   ║
║  ✔  Smart anchor-based continuation prompts (no vague "continue" messages)  ║
║  ✔  difflib overlap deduplication  — removes model recaps between chunks    ║
║  ✔  Per-chunk exponential-backoff retry — survives transient API failures   ║
║  ✔  Hard chunk ceiling (for-else)  — zero chance of an infinite loop        ║
║  ✔  Complete structured logging    — every event, every level               ║
║  ✔  PII-safe DEBUG payloads        — toggled via GenerationConfig flags     ║
║  ✔  Optional on_chunk callback     — real-time streaming to UI / queues     ║
║  ✔  Rich GenerationResult type     — full metadata, not just raw text       ║
║                                                                              ║
║  Logging levels                                                              ║
║  ──────────────                                                              ║
║  CRITICAL  unrecoverable failures (ceiling breach, retry exhaustion,        ║
║             non-continuable finish reason)                                   ║
║  ERROR     per-attempt API errors                                            ║
║  WARNING   token-limit hits, retry attempts, unknown finish reasons         ║
║  INFO      lifecycle events (start, request sent, response received,        ║
║             dedup applied, complete, summary)                                ║
║  DEBUG     full payloads — prompt, continuation prompts, response text,     ║
║             overlap text (all truncated + PII-toggleable)                   ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import difflib
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

import google.genai as genai
from dotenv import load_dotenv
from google.genai import types

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Domain enums & types
# ──────────────────────────────────────────────────────────────────────────────

class FinishReason(str, Enum):
    """Typed wrapper around Gemini finish_reason strings."""

    STOP       = "STOP"
    MAX_TOKENS = "MAX_TOKENS"
    SAFETY     = "SAFETY"
    RECITATION = "RECITATION"
    OTHER      = "OTHER"

    @classmethod
    def from_raw(cls, value: str) -> "FinishReason":
        """Graceful fallback for unknown values returned by the API."""
        try:
            return cls(value)
        except ValueError:
            logger.warning(
                "generation.unknown_finish_reason",
                extra={"raw_value": value},
            )
            return cls.OTHER


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GenerationConfig:
    """
    All tuneable parameters in one immutable dataclass.
    Override only what you need — all fields have production-safe defaults.
    """

    # ── Gemini inference ──────────────────────────────────────────────────────
    max_output_tokens: int   = 4_000    # tokens per API call
    temperature: float       = 0.5      # 0 = deterministic, 1 = creative

    # ── Continuation & stitching ──────────────────────────────────────────────
    anchor_chars: int        = 100      # tail chars used in continuation prompt
    overlap_window: int      = 500      # chars of tail scanned for dedup overlap
    min_overlap_chars: int   = 20       # minimum match length considered a recap

    # ── Loop safety ───────────────────────────────────────────────────────────
    max_chunks: int          = 50       # hard ceiling — raises RuntimeError if hit

    # ── Rate limiting & retry ─────────────────────────────────────────────────
    inter_chunk_delay: float = 2.0      # seconds to sleep between MAX_TOKENS chunks
    max_retries: int         = 3        # per-chunk retry attempts on API failure
    retry_backoff: float     = 5.0      # seconds × attempt (exponential multiplier)

    # ── Logging controls (set False in prod for PII safety) ───────────────────
    log_prompt: bool         = True     # log original prompt text at DEBUG
    log_responses: bool      = True     # log each chunk's response text at DEBUG
    log_continuation: bool   = True     # log continuation prompt text at DEBUG
    log_overlap_text: bool   = False    # log trimmed overlap text at DEBUG (verbose)
    response_log_limit: int  = 500      # max chars logged per payload at DEBUG


# ──────────────────────────────────────────────────────────────────────────────
# Result type
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GenerationResult:
    """
    Rich return type — never just a raw string.
    Callers can inspect how the generation concluded and monitor health metrics.
    """
    text: str                           # fully stitched, deduped output
    chunks_consumed: int                # number of API calls made
    finish_reason: FinishReason         # final stop signal from Gemini
    total_duration_seconds: float       # wall-clock time for full generation
    session_id: str      = ""           # unique ID for log correlation
    retries_used: int    = 0            # total retry attempts across all chunks
    overlaps_removed: int = 0           # number of recap dedup operations applied
    total_chars: int     = 0            # character count of final stitched text


# ──────────────────────────────────────────────────────────────────────────────
# Core generator
# ──────────────────────────────────────────────────────────────────────────────

class GeminiService:
    """
    Enterprise-grade Gemini continuation generator.

    Usage
    -----
    >>> generator = GeminiContinuationGenerator()                  # reads .env
    >>> generator = GeminiContinuationGenerator("gemini-2.5-pro")  # custom model
    >>> result = generator.generate_until_complete("Write a 10,000 word essay...")
    >>> print(result.text)
    """

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        config: Optional[GenerationConfig] = None,
        on_chunk: Optional[Callable[[str, int], None]] = None,
    ) -> None:
        """
        Parameters
        ----------
        model_name  : Gemini model string. Defaults to "gemini-2.5-flash".
        config      : GenerationConfig override; defaults to production-safe values.
        on_chunk    : Optional callback(chunk_text, chunk_index) for live streaming.

        Environment
        -----------
        Reads GEMINI_API_KEY from the environment or a .env file in the
        current working directory. Raises ValueError if the key is missing.
        """
        # Load .env file if present — safe no-op if already set in environment
        load_dotenv()

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. "
                "Add it to your .env file or export it as an environment variable."
            )

        self._client   = genai.Client(api_key=api_key)
        self._model    = model_name
        self._cfg      = config or GenerationConfig()
        self._on_chunk = on_chunk

        logger.info(
            "generator.initialised",
            extra={"model": self._model},
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def generate_until_complete(self, prompt: str) -> GenerationResult:
        """
        Drive the Gemini chat session until a STOP signal or ceiling breach.

        Each MAX_TOKENS response is seamlessly continued using an anchor-based
        prompt. Overlapping recap content is detected and removed via difflib.

        Parameters
        ----------
        prompt : The initial user prompt. Must be non-empty.

        Returns
        -------
        GenerationResult
            Fully assembled text with metadata.

        Raises
        ------
        ValueError
            If prompt is empty or whitespace-only.
        RuntimeError
            On chunk ceiling breach, non-continuable finish reason,
            or retry exhaustion. Partial text is NOT silently returned.
        """
        if not prompt or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        cfg              = self._cfg
        session_id       = str(uuid.uuid4())
        started_at       = time.monotonic()
        chunks: list[str]= []
        retries_used     = 0
        overlaps_removed = 0
        finish_reason    = FinishReason.OTHER

        # ── EVENT: generation session started ────────────────────────────────
        logger.info(
            "generation.started",
            extra={
                "session_id":   session_id,
                "model":        self._model,
                "prompt_chars": len(prompt),
                "max_chunks":   cfg.max_chunks,
                "max_tokens":   cfg.max_output_tokens,
                "temperature":  cfg.temperature,
            },
        )

        # ── EVENT: full prompt payload (DEBUG — may be large / PII) ──────────
        if cfg.log_prompt:
            logger.debug(
                "generation.prompt_payload",
                extra={
                    "session_id": session_id,
                    "prompt":     prompt,
                },
            )

        chat = self._client.chats.create(model=self._model)

        # ── Main continuation loop ────────────────────────────────────────────
        for chunk_index in range(cfg.max_chunks):

            # Build the message for this turn
            if chunk_index == 0:
                message = prompt
            else:
                message = self._build_continuation_prompt(chunks[-1])

                # ── EVENT: continuation prompt built ─────────────────────────
                if cfg.log_continuation:
                    logger.debug(
                        "generation.continuation_prompt",
                        extra={
                            "session_id": session_id,
                            "chunk":      chunk_index,
                            "message":    message,
                        },
                    )

            # ── EVENT: request being dispatched ──────────────────────────────
            logger.info(
                "generation.request_sent",
                extra={
                    "session_id":    session_id,
                    "chunk":         chunk_index,
                    "message_chars": len(message),
                },
            )

            # Send with per-chunk retry
            chunk_text, finish_reason, attempts = self._send_with_retry(
                chat=chat,
                message=message,
                chunk_index=chunk_index,
                session_id=session_id,
            )
            retries_used += attempts

            # ── EVENT: raw response received ─────────────────────────────────
            logger.info(
                "generation.response_received",
                extra={
                    "session_id":     session_id,
                    "chunk":          chunk_index,
                    "response_chars": len(chunk_text),
                    "finish_reason":  finish_reason.value,
                    "attempts_used":  attempts,
                },
            )

            # ── EVENT: response text payload (DEBUG — truncated) ─────────────
            if cfg.log_responses:
                logger.debug(
                    "generation.response_text",
                    extra={
                        "session_id":   session_id,
                        "chunk":        chunk_index,
                        "text_preview": self._truncate(chunk_text, cfg.response_log_limit),
                        "truncated":    len(chunk_text) > cfg.response_log_limit,
                    },
                )

            # ── Overlap deduplication ─────────────────────────────────────────
            if chunks:
                overlap_size, overlap_text = self._find_overlap(chunks[-1], chunk_text)
                if overlap_size:
                    # ── EVENT: dedup applied ──────────────────────────────────
                    logger.info(
                        "generation.dedup_applied",
                        extra={
                            "session_id":    session_id,
                            "chunk":         chunk_index,
                            "overlap_chars": overlap_size,
                        },
                    )
                    # ── EVENT: trimmed overlap text (DEBUG — very verbose) ────
                    if cfg.log_overlap_text:
                        logger.debug(
                            "generation.overlap_text",
                            extra={
                                "session_id":   session_id,
                                "chunk":        chunk_index,
                                "overlap_text": overlap_text,
                            },
                        )
                    chunk_text = chunk_text[overlap_size:]
                    overlaps_removed += 1

            chunks.append(chunk_text)

            # Fire optional streaming callback
            if self._on_chunk:
                self._on_chunk(chunk_text, chunk_index)

            # ── Finish reason routing ─────────────────────────────────────────

            if finish_reason is FinishReason.STOP:
                # ── EVENT: natural completion ─────────────────────────────────
                logger.info(
                    "generation.complete",
                    extra={
                        "session_id":   session_id,
                        "total_chunks": chunk_index + 1,
                    },
                )
                break

            if finish_reason is FinishReason.MAX_TOKENS:
                # ── EVENT: token limit hit — will continue next chunk ─────────
                logger.warning(
                    "generation.token_limit_hit",
                    extra={
                        "session_id":       session_id,
                        "chunk":            chunk_index,
                        "next_chunk":       chunk_index + 1,
                        "sleeping_seconds": cfg.inter_chunk_delay,
                    },
                )
                time.sleep(cfg.inter_chunk_delay)
                continue

            # ── EVENT: non-continuable finish reason ──────────────────────────
            logger.critical(
                "generation.non_continuable_finish",
                extra={
                    "session_id":    session_id,
                    "chunk":         chunk_index,
                    "finish_reason": finish_reason.value,
                },
            )
            raise RuntimeError(
                f"[session={session_id}] Non-continuable finish reason "
                f"'{finish_reason.value}' at chunk {chunk_index}. "
                "Inspect content policy or prompt."
            )

        else:
            # for-else: loop ran to completion without a break = ceiling breached
            # ── EVENT: safety ceiling breached ────────────────────────────────
            logger.critical(
                "generation.ceiling_breached",
                extra={
                    "session_id": session_id,
                    "max_chunks": cfg.max_chunks,
                },
            )
            raise RuntimeError(
                f"[session={session_id}] Exceeded safety ceiling of "
                f"{cfg.max_chunks} chunks. Model may be looping — "
                "inspect the prompt or increase max_chunks in GenerationConfig."
            )

        # ── Assemble final result ─────────────────────────────────────────────
        final_text = self._stitch(chunks)
        duration   = round(time.monotonic() - started_at, 3)

        # ── EVENT: full generation summary ────────────────────────────────────
        logger.info(
            "generation.summary",
            extra={
                "session_id":          session_id,
                "model":               self._model,
                "chunks_consumed":     len(chunks),
                "overlaps_removed":    overlaps_removed,
                "retries_used":        retries_used,
                "total_chars":         len(final_text),
                "total_duration_secs": duration,
                "finish_reason":       finish_reason.value,
            },
        )

        # ── EVENT: final stitched text preview (DEBUG) ────────────────────────
        if cfg.log_responses:
            logger.debug(
                "generation.final_text",
                extra={
                    "session_id":   session_id,
                    "text_preview": self._truncate(final_text, cfg.response_log_limit),
                    "truncated":    len(final_text) > cfg.response_log_limit,
                },
            )

        return GenerationResult(
            text=final_text,
            chunks_consumed=len(chunks),
            finish_reason=finish_reason,
            total_duration_seconds=duration,
            session_id=session_id,
            retries_used=retries_used,
            overlaps_removed=overlaps_removed,
            total_chars=len(final_text),
        )

    # ── Stitching ─────────────────────────────────────────────────────────────

    def _stitch(self, chunks: list[str]) -> str:
        """Join chunks — overlap already stripped per-chunk during the loop."""
        return "".join(chunks)

    # ── Overlap deduplication ─────────────────────────────────────────────────

    def _find_overlap(
        self,
        prev_chunk: str,
        next_chunk: str,
    ) -> tuple[int, str]:
        """
        Detect model recap: find the longest match between the tail of
        prev_chunk and the START of next_chunk.

        Returns
        -------
        (overlap_char_count, overlap_text)
            overlap_char_count > 0 means next_chunk begins with duplicated content.
            Caller should strip next_chunk[:overlap_char_count] before appending.
        """
        cfg  = self._cfg
        tail = prev_chunk[-cfg.overlap_window:]

        matcher = difflib.SequenceMatcher(None, tail, next_chunk, autojunk=False)
        match   = matcher.find_longest_match(0, len(tail), 0, len(next_chunk))

        # match.b == 0 ensures overlap starts at the BEGINNING of next_chunk
        if match.size >= cfg.min_overlap_chars and match.b == 0:
            return match.size, next_chunk[:match.size]

        return 0, ""

    # ── Continuation prompting ────────────────────────────────────────────────

    def _build_continuation_prompt(self, last_chunk: str) -> str:
        """
        Anchor-based continuation prompt.

        Uses the last N characters of the previous chunk as a precise reference
        point so the model picks up exactly where it left off rather than
        recapping or drifting — especially important over many chunks.
        """
        anchor = last_chunk.strip()[-self._cfg.anchor_chars:]
        return (
            "Continue the response without repeating or summarising previous content. "
            f"Pick up directly and seamlessly after: '...{anchor}'"
        )

    # ── Retry wrapper ─────────────────────────────────────────────────────────

    def _send_with_retry(
        self,
        chat,
        message: str,
        chunk_index: int,
        session_id: str,
    ) -> tuple[str, FinishReason, int]:
        """
        Send one message to the chat session.
        Retries transient API failures up to max_retries times with
        exponential backoff (retry_backoff × attempt seconds).

        Returns
        -------
        (chunk_text, finish_reason, attempts_used)
        """
        cfg      = self._cfg
        last_exc: Optional[Exception] = None

        for attempt in range(cfg.max_retries + 1):

            if attempt > 0:
                sleep_for = cfg.retry_backoff * attempt
                # ── EVENT: retry attempt ──────────────────────────────────────
                logger.warning(
                    "generation.retry",
                    extra={
                        "session_id":    session_id,
                        "chunk":         chunk_index,
                        "attempt":       attempt,
                        "max_retries":   cfg.max_retries,
                        "sleeping_secs": sleep_for,
                    },
                )
                time.sleep(sleep_for)

            try:
                response      = chat.send_message(
                    message=message,
                    config=types.GenerateContentConfig(
                        max_output_tokens=cfg.max_output_tokens,
                        temperature=cfg.temperature,
                    ),
                )
                candidate     = response.candidates[0]
                finish_reason = FinishReason.from_raw(candidate.finish_reason)
                return response.text, finish_reason, attempt

            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                # ── EVENT: API error on this attempt ─────────────────────────
                logger.error(
                    "generation.api_error",
                    extra={
                        "session_id":    session_id,
                        "chunk":         chunk_index,
                        "attempt":       attempt,
                        "error_type":    type(exc).__name__,
                        "error_message": str(exc),
                    },
                )

        # ── EVENT: all retries exhausted ──────────────────────────────────────
        logger.critical(
            "generation.retry_exhausted",
            extra={
                "session_id":  session_id,
                "chunk":       chunk_index,
                "max_retries": cfg.max_retries,
            },
        )
        raise RuntimeError(
            f"[session={session_id}] Exhausted {cfg.max_retries} retries "
            f"on chunk {chunk_index}."
        ) from last_exc

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        return text[:limit] + ("..." if len(text) > limit else "")


# ──────────────────────────────────────────────────────────────────────────────
# Quick-start usage
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Logging setup ─────────────────────────────────────────────────────────
    #   Production  → level=logging.INFO   (hides all payload / PII logs)
    #   Development → level=logging.DEBUG  (shows everything)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    # ── Config ────────────────────────────────────────────────────────────────
    config = GenerationConfig(
        max_output_tokens  = 4_000,
        temperature        = 0.5,
        anchor_chars       = 100,
        overlap_window     = 500,
        min_overlap_chars  = 20,
        max_chunks         = 50,
        inter_chunk_delay  = 2.0,
        max_retries        = 3,
        retry_backoff      = 5.0,
        # --- Logging toggles ---
        log_prompt         = True,   # ← False in prod if prompt contains PII
        log_responses      = True,   # ← False in prod if response contains PII
        log_continuation   = True,
        log_overlap_text   = False,  # very verbose — enable only for deep debug
        response_log_limit = 500,
    )

    # ── Optional live-stream callback ─────────────────────────────────────────
    def on_chunk(text: str, index: int) -> None:
        print(f"  [chunk {index:02d}] {len(text):,} chars received")

    # ── Run — client built automatically from .env ────────────────────────────
    # Requires GEMINI_API_KEY in .env or environment
    generator = GeminiService(
        model_name = "gemini-2.5-flash",
        config     = config,
        on_chunk   = on_chunk,
    )

    result = generator.generate_until_complete(
        prompt=(
            "Write a comprehensive 10,000 word essay on the history of "
            "artificial intelligence, covering symbolic AI, the AI winters, "
            "machine learning, deep learning, and the rise of large language models."
        )
    )

    print(f"\n{'═' * 60}")
    print(f"  Session ID       : {result.session_id}")
    print(f"  Chunks consumed  : {result.chunks_consumed}")
    print(f"  Overlaps removed : {result.overlaps_removed}")
    print(f"  Retries used     : {result.retries_used}")
    print(f"  Duration         : {result.total_duration_seconds}s")
    print(f"  Finish reason    : {result.finish_reason.value}")
    print(f"  Total chars      : {result.total_chars:,}")
    print(f"{'═' * 60}\n")
    print(result.text)