"""
Orchestrator: serializes inference jobs onto the local VL model (a single
GPU instance) via one worker, then ships each result back to its requester
over the edge socket.
"""

from __future__ import annotations
from typing import Optional, TYPE_CHECKING
import asyncio
import logging

import config

if TYPE_CHECKING:
    from server import InferenceJob
    from Persona.Character import Character
    from edge_client import EdgeClient

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Owns the inference queue and its single worker.

    Jobs go in via submit(); each one is turned into a persona prompt
    through `character` (which round-trips to Fuli for memories/emotion),
    run through the VL model, and the result is sent as "inference_result"
    (and, on success, "tts_request") over the edge socket. Both message
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

    async def submit(self, job: "InferenceJob") -> None:
        """Enqueue a job for the worker to pick up."""
        await self._queue.put(job)

    async def check_downstream_servers(self) -> dict[str, bool]:
        """
        Startup liveness sweep: a precise check for the VL model (validates
        it's actually loaded, not just that a port is open) plus a generic
        TCP-reachability check for every other microservice in config.servers.
        Logs the results and never raises — a downstream server being down
        shouldn't block the orchestrator from booting.
        """
        vl_ok, vl_reason = await asyncio.to_thread(self.character.vl_handler.check_server)
        if vl_ok:
            logger.info(f"VL server reachable ({self.character.vl_handler.alias})")
        else:
            logger.warning(f"VL server not reachable: {vl_reason}")

        results = await config.test_servers()
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
            try:
                await self._process(job)
            except Exception:
                logger.exception("[Orchestrator] job failed (request_id=%s)", job.request_id)
            finally:
                self._queue.task_done()

    async def _process(self, job: "InferenceJob") -> None:
        # get_character_prompt() round-trips to Fuli (memories + emotion),
        # then VisionLangHandler.inference() is a blocking `requests.post`
        # call — both are blocking, so both run off the event loop.
        system_prompt = await asyncio.to_thread(
            self.character.get_character_prompt,
            user_name=job.user_name or job.from_role,
            user_input=job.text,
        )

        success, text = await asyncio.to_thread(
            self.character.vl_handler.inference,
            user_prompt=job.text,
            system_prompt=system_prompt,
            image_base64=job.image_base64,
        )

        if success:
            # Closes the memory buffer __call_Fuli__ opened for this user_name;
            # Fuli refuses their *next* request until this runs. Never let a
            # save failure swallow a response the user already got a reply to.
            try:
                await asyncio.to_thread(
                    self.character.save_turn,
                    job.user_name or job.from_role,
                    text,
                )
            except Exception:
                logger.exception("[Orchestrator] Fuli save_turn failed (request_id=%s)", job.request_id)

        payload = {
            "request_id": job.request_id,
            "status": "success" if success else "error",
            "output_text": text if success else None,
            "error": None if success else text,
        }
        # no `to` here: the hub's routing table fixes (Orchestrator, "inference_result") -> Client,
        # so it always fans out to whichever Client(s) are connected regardless of job.from_role
        await self.edge.send_routed("inference_result", payload)

        if success:
            # likewise fixed by the hub: (Orchestrator, "tts_request") -> Tts
            await self.edge.send_routed("tts_request", {"request_id": job.request_id, "text": text})
