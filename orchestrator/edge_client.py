from typing import Callable, Awaitable, Union
import asyncio
import contextlib
import json
import logging
import time
import websockets

logger = logging.getLogger(__name__)

# After the hub link drops, the first reconnect attempt waits RECONNECT_BASE_DELAY seconds and each later one
# twice as long as the one before, capped at RECONNECT_MAX_DELAY.
RECONNECT_BASE_DELAY = 1.0
RECONNECT_MAX_DELAY = 30.0
# A link that stayed up this long counts as healthy, so the next drop starts over at the base delay. One that
# dies sooner (the hub accepts our register, then drops us again: duplicate role, crash loop) keeps the backoff
# growing instead of being re-dialled once a second.
RECONNECT_STABLE_SECONDS = 10.0


class EdgeClient:
    def __init__(self, edge_url: str, role: str) -> None:
        self.edge_url: str = edge_url
        self.role: str = role
        self.ws = None
        self._listen_task: asyncio.Task | None = None

    async def _open(self) -> str:
        """
        Dials the hub and registers our role; returns the role the hub acked. Both the first connect and every
        reconnect go through here, so the register message is always re-sent.

        self.ws is only set once the hub has acked: until then connected is False and the send_* methods refuse,
        and a failed or cancelled attempt closes its socket instead of leaving it behind.
        """
        ws = await websockets.connect(self.edge_url, max_size=None)
        try:
            await ws.send(json.dumps({
                "type": "register",
                "role": self.role,
            }))
            ack = json.loads(await ws.recv())
            role = ack.get("role")
        except BaseException:
            await self._close_quietly(ws)
            raise
        self.ws = ws
        return role

    async def connect(self):
        role = await self._open()
        print(f"[EdgeClient] Connected to edge, registered as: {role}")
 
    @property
    def connected(self) -> bool:
        """
        True while the websocket to the edge hub is open and registered (the TUI status bar shows this).
        False from the moment the link drops until the listener has reconnected it.
        """
        state = getattr(self.ws, "state", None)
        return state is not None and state.name == "OPEN"

    def _require_ws(self):
        ws = self.ws
        if ws is None:
            # not a silent no-op: the caller (an inference_result / tts_request) would think it was delivered
            raise ConnectionError("not connected to the edge hub")
        return ws

    async def send_json(self, data: dict):
        await self._require_ws().send(json.dumps(data))
 
    async def send_bytes(self, data: bytes):
        await self._require_ws().send(data)
 
    async def send_routed(self, msg_type: str, payload: dict, to: str = None):
        msg = {"type": msg_type, "payload": payload}
        if to:
            msg["to"] = to
        await self.send_json(msg)
 
    def start_listener(self, handler: Callable[[Union[dict, bytes]], Awaitable[None]]):
        self._listen_task = asyncio.create_task(self._listen_loop(handler))
 
    async def _listen_loop(self, handler):
        """
        Hands every hub message to handler for as long as the client lives. When the link ends - for any reason,
        including the hub closing it cleanly - it logs why, then reconnects with capped exponential backoff and
        carries on, so the orchestrator never has to be restarted to get the edge back.
        """
        delay = RECONNECT_BASE_DELAY
        while True:
            ws = self.ws
            up_since = time.monotonic()
            reason = await self._receive(ws, handler)

            # Over, whatever the reason. Drop the socket right away: connected and the send_* methods go by
            # self.ws, so a dead socket must not stay in it (a clean close by the hub ends the iteration silently).
            self.ws = None
            if time.monotonic() - up_since >= RECONNECT_STABLE_SECONDS:
                delay = RECONNECT_BASE_DELAY
            logger.warning("Edge listener ended: %s; reconnecting to %s in %gs", reason, self.edge_url, delay)
            await self._close_quietly(ws)
            delay = await self._reconnect(delay)

    async def _receive(self, ws, handler) -> str:
        """Feeds every frame from ws to handler until the link ends. Returns why it ended, for the log."""
        try:
            while True:
                msg = await ws.recv()
                try:
                    await handler(json.loads(msg) if isinstance(msg, str) else msg)
                except websockets.ConnectionClosed:
                    raise
                except Exception:
                    # a malformed frame or a handler bug costs that one message, not the connection
                    logger.exception("Edge message handler failed; message dropped")
        except websockets.ConnectionClosedOK as exc:
            return f"hub closed the connection ({exc})"
        except websockets.ConnectionClosed as exc:
            return f"connection lost ({exc})"
        except Exception as exc:
            logger.exception("Edge listener failed unexpectedly")
            return f"unexpected {type(exc).__name__}: {exc}"

    async def _reconnect(self, delay: float) -> float:
        """
        Dials the hub until it takes us back: waits delay before the first attempt and twice as long after each
        failure, never more than RECONNECT_MAX_DELAY. Returns the delay the next outage starts from if this
        link turns out not to be stable (see RECONNECT_STABLE_SECONDS).
        """
        attempts = 0
        while True:
            await asyncio.sleep(delay)
            delay = min(delay * 2, RECONNECT_MAX_DELAY)
            attempts += 1
            try:
                role = await self._open()
            except Exception as exc:
                logger.warning("Edge reconnect failed (%s: %s); retrying in %gs", type(exc).__name__, exc, delay)
                continue
            logger.info("Edge reconnected after %d attempt(s), registered as: %s", attempts, role)
            return delay

    @staticmethod
    async def _close_quietly(ws) -> None:
        if ws is not None:
            with contextlib.suppress(Exception):
                await ws.close()
 
    async def close(self):
        task, self._listen_task = self._listen_task, None
        if task is not None:
            task.cancel()
            # let it unwind before closing the socket: it may be mid-reconnect and must not dial again
            await asyncio.wait([task])
        ws, self.ws = self.ws, None
        await self._close_quietly(ws)
