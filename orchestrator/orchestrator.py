"""
Orchestrator: serializes inference jobs onto the local VL model (a single
GPU instance) via one worker, then ships each result back to its requester
over the edge socket.
"""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING
import asyncio
import logging
import time

import config

if TYPE_CHECKING:
    from server import InferenceJob
    from Persona.Character import Character
    from edge_client import EdgeClient

logger = logging.getLogger(__name__)

# The TUI's verbose view shows the DEBUG records below. One record never carries more than this many
# characters of a prompt / reply / reasoning text.
DEBUG_TEXT_LIMIT = 6000


def _clip(text: Optional[str], limit: int = DEBUG_TEXT_LIMIT) -> str:
    text = text or ""
    return text if len(text) <= limit else f"{text[:limit]}\n... [{len(text) - limit} more chars]"


def _debug_vl_result(rid: str, success: bool, text: str, meta: dict) -> None:
    """DEBUG trace of one VL call; `meta` is the third return value of VisionLangHandler.inference()."""
    usage = meta.get("usage") or {}
    logger.debug(
        "[%s] VL %s in %s ms  model=%s  finish=%s  tokens: prompt=%s completion=%s reasoning=%s cached=%s",
        rid, "ok" if success else "FAILED", meta.get("latency_ms"), meta.get("model"), meta.get("finish_reason"),
        usage.get("prompt_tokens"), usage.get("completion_tokens"), usage.get("reasoning_tokens"),
        usage.get("cached_tokens"),
    )
    if meta.get("reasoning_content"):
        logger.debug("[%s] VL reasoning:\n%s", rid, _clip(meta["reasoning_content"]))
    logger.debug("[%s] VL %s:\n%s", rid, "reply" if success else "error", _clip(text))


class Orchestrator:
    """
    Owns the inference queue and its single worker.

    Jobs go in via submit(); each one is turned into a persona prompt
    through `character` (which round-trips to Fuli for memories/emotion),
    run through the VL model, and the result is either handed back on
    the job's `reply` future (HTTP callers, e.g. Discord) or sent as
    "inference_result" (and, on success, "tts_request") over the edge socket. Both message
    types are implicitly routed by the edge hub's static routing table
    (Orchestrator -> Client / Tts), so no explicit `to` is needed.
    """

    def __init__(
        self,
        character: "Character",
        edge: "EdgeClient",
    ) -> None:
        self.character = character
        self.edge = edge

        self._queue: "asyncio.Queue[InferenceJob]" = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None

        # read-only bookkeeping for dashboards (see snapshot()); nothing in the
        # job path reads these back
        self._current_job: Optional["InferenceJob"] = None
        self._current_started: float = 0.0
        self._jobs_ok = 0
        self._jobs_failed = 0
        self._last_duration: Optional[float] = None

    async def submit(self, job: "InferenceJob") -> None:
        """Enqueue a job for the worker to pick up."""
        await self._queue.put(job)

    def snapshot(self) -> dict:
        """
        Cheap, non-blocking view of the queue for dashboards (the TUI):
        waiting jobs, the job the worker is on right now, and ok/failed totals
        since boot. "failed" covers both a VL/Fuli error reply and a job that raised.
        """
        job = self._current_job
        return {
            "queued": self._queue.qsize(),
            "in_flight": None if job is None else {
                "request_id": job.request_id,
                "user_name": job.user_name,
                "platform": job.platform,
                "elapsed_s": time.monotonic() - self._current_started,
            },
            "jobs_ok": self._jobs_ok,
            "jobs_failed": self._jobs_failed,
            "last_duration_s": self._last_duration,
        }

    async def probe_downstream_servers(self) -> tuple[bool, str, dict[str, bool]]:
        """
        The liveness sweep without any logging: (VL model ok, the VL check's
        failure reason or "", {config.servers name: reachable}).
        check_downstream_servers() is this plus the startup log lines; the TUI
        re-runs this one on a timer, so it must stay silent.
        """
        vl_ok, vl_reason = await asyncio.to_thread(self.character.vl_handler.check_server)
        results = await config.test_servers()
        return vl_ok, vl_reason, results

    async def check_downstream_servers(self) -> dict[str, bool]:
        """
        Startup liveness sweep: a precise check for the VL model (validates
        it's actually loaded, not just that a port is open) plus a generic
        TCP-reachability check for every other microservice in config.servers.
        Logs the results and never raises — a downstream server being down
        shouldn't block the orchestrator from booting.
        """
        vl_ok, vl_reason, results = await self.probe_downstream_servers()
        if vl_ok:
            logger.info(f"VL server reachable ({self.character.vl_handler.alias})")
        else:
            logger.warning(f"VL server not reachable: {vl_reason}")

        for name, alive in results.items():
            logger.info(f"{name}: {'alive' if alive else 'unreachable'}")

        return {"vl_model": vl_ok, **results}

    def start(self) -> None:
        """Spawn the worker loop. Safe to call once per lifespan."""
        if self._worker_task is not None:
            return
        self._worker_task = asyncio.create_task(self._worker_loop())

    async def stop(self) -> None:
        """Cancel the worker. Jobs still sitting in the queue are dropped."""
        if self._worker_task is None:
            return
        self._worker_task.cancel()
        try:
            await self._worker_task
        except asyncio.CancelledError:
            pass
        self._worker_task = None

    async def _worker_loop(self) -> None:
        # Single worker on purpose: the VL model is one local GPU instance,
        # so jobs are processed strictly one at a time to avoid contention/OOM.
        while True:
            job = await self._queue.get()
            self._current_job, self._current_started = job, time.monotonic()
            outcome = "cancelled"  # only survives if the worker task itself is cancelled mid-job
            try:
                if await self._process(job):
                    self._jobs_ok += 1
                    outcome = "ok"
                else:
                    self._jobs_failed += 1
                    outcome = "failed"
            except Exception as e:
                self._jobs_failed += 1
                outcome = "raised"
                logger.exception("[Orchestrator] job failed (request_id=%s)", job.request_id)
                # an HTTP caller is awaiting this future — don't leave it hanging until its timeout
                if job.reply is not None and not job.reply.done():
                    job.reply.set_exception(e)
            finally:
                self._last_duration = time.monotonic() - self._current_started
                self._current_job = None
                self._queue.task_done()
                logger.debug("[%s] finished: %s, %.2fs in total", str(job.request_id)[:8], outcome, self._last_duration)

    async def _process(self, job: "InferenceJob") -> bool:
        """Runs one job end to end. Returns whether inference succeeded (the worker's ok/failed tally)."""
        # The DEBUG trace in here is what the TUI's verbose view shows; with DEBUG off it costs nothing.
        debug = logger.isEnabledFor(logging.DEBUG)
        rid = str(job.request_id)[:8]
        started = time.monotonic()
        if debug:
            logger.debug(
                "[%s] start: %s @ %s  user_key=%s  scope_key=%s  image=%s",
                rid, job.user_name, job.platform, job.user_key, job.scope_key,
                f"{len(job.image_base64)} base64 chars" if job.image_base64 else "no",
            )
            logger.debug("[%s] user input:\n%s", rid, _clip(job.text))
        # get_character_prompt() round-trips to Fuli (memories + emotion),
        # then VisionLangHandler.inference() is a blocking `requests.post`
        # call — both are blocking, so both run off the event loop.
        system_prompt = await asyncio.to_thread(
            self.character.get_character_prompt,
            user_key=job.user_key,
            scope_key=job.scope_key,
            user_name=job.user_name,
            user_input=job.text,
            platform=job.platform,
        )
        if debug:
            logger.debug(
                "[%s] system prompt, %d chars, ready after %.2fs:\n%s",
                rid, len(system_prompt), time.monotonic() - started, _clip(system_prompt),
            )

        success, text, llm_meta = await asyncio.to_thread(
            self.character.vl_handler.inference,
            user_prompt=job.text,
            system_prompt=system_prompt,
            image_base64=job.image_base64,
            request_id=job.request_id,
        )
        if debug:
            _debug_vl_result(rid, success, text, llm_meta)

        memory_payload = None
        if success:
            # Closes the memory buffer __call_Fuli__ opened for this user;
            # Fuli refuses their *next* request until this runs. Never let a
            # save failure swallow a response the user already got a reply to.
            try:
                save_result = await asyncio.to_thread(
                    self.character.save_turn,
                    job.user_key,
                    job.scope_key,
                    job.user_name,
                    text,
                )
                # to_client_payload() strips `query` (a verbose echo of our
                # own Fuli request) and the hashed `user_id` — neither
                # should ever reach the frontend.
                memory_payload = save_result.to_client_payload()
            except Exception:
                logger.exception("[Orchestrator] Fuli save_turn failed (request_id=%s)", job.request_id)
            if debug and memory_payload is not None:
                analysis = (memory_payload.get("emotion_analysis") or {}).get("deltaEGO_analysis") or {}
                logger.debug(
                    "[%s] Fuli save ok  metadata=%s  emotion=%s",
                    rid, memory_payload.get("metadata"), analysis.get("emotion_term"),
                )

        payload = {
            "request_id": job.request_id,
            "status": "success" if success else "error",
            "output_text": text if success else None,
            "error": None if success else text,
            "memory": memory_payload,
        }

        # HTTP callers (Discord) get the result straight back and never touch
        # the edge: the hub would fan it out to every Unity Client and voice it via Tts
        if job.reply is not None:
            if not job.reply.done():
                job.reply.set_result(payload)
            if debug:
                logger.debug("[%s] answered the HTTP caller (%s)", rid, payload["status"])
            return success

        # no `to` here: the hub's routing table fixes (Orchestrator, "inference_result") -> Client,
        # so it always fans out to whichever Client(s) are connected regardless of job.from_role
        await self.edge.send_routed("inference_result", payload)
        if debug:
            logger.debug("[%s] sent inference_result to the edge hub (%s)", rid, payload["status"])

        if success:
            # likewise fixed by the hub: (Orchestrator, "tts_request") -> Tts
            await self.edge.send_routed("tts_request", {"request_id": job.request_id, "text": text, "speaker" : "reminh"})  #TODO: need to be fixed.
            if debug:
                logger.debug("[%s] sent tts_request to the edge hub (%d chars)", rid, len(text))

        return success
