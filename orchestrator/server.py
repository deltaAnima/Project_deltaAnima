"""
external libs
"""
import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from dotenv import load_dotenv
from typing import Optional
import asyncio
import logging
import time
import uuid
import os

load_dotenv()


"""
    custom libs
"""
from Server.Pydantic_frame import(
    DiscordBotTextRequest,
)
from VisionLangHandler import VisionLangHandler
from edge_client import EdgeClient
from orchestrator import Orchestrator
from Persona.Character import Character
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("reminh.server")

# ---------------------
# 1. Waking up.....
# ---------------------
import os
import pwd
print(f"Welcome Mr/Mrs {pwd.getpwuid(os.getuid()).pw_name}!!!\nWaking up Reminh now......")

# ---------------------
# 2. initalizing handlers
# ---------------------
print("Step 1. Initializing handlers........")
VL_handler = VisionLangHandler(
    address=config.vl_model.address,
    port=config.vl_model.port,
    alias=config.vl_model.alias,
    temperature=config.vl_model.temperature,
    max_token=config.vl_model.max_token,
    top_p=config.vl_model.top_p,
    model_full=config.vl_model.model_full,
)

edge = EdgeClient(
    edge_url=config.edge.url,
    role=config.edge.role,
)

# TODO: make the active character configurable once more than one exists
character = Character(chara_name="Reminh", vl_handler=VL_handler)



class EdgeRPC:
    def __init__(self, edge: EdgeClient, timeout: float = 10.0):
        self.edge = edge
        self.timeout = timeout
        self._pending: dict[str, asyncio.Future] = {}

    async def request(self, msg_type: str, payload: dict, to: Optional[str] = None, timeout: Optional[float] = None) -> dict:
        request_id = str(uuid.uuid4())
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending[request_id] = future
        try:
            if to is not None:
                await self.edge.send_routed(msg_type, {**payload, "request_id": request_id}, to=to)
            else:
                await self.edge.send_routed(msg_type, {**payload, "request_id": request_id})
            return await asyncio.wait_for(future, timeout=timeout or self.timeout)
        finally:
            self._pending.pop(request_id, None)

    def resolve(self, request_id: str, payload: dict):
        future = self._pending.get(request_id)
        if future and not future.done():
            future.set_result(payload)


edge_rpc = EdgeRPC(edge)

orchestrator = Orchestrator(
    character=character,
    edge=edge,
)


@dataclass
class InferenceJob:
    text: str
    request_id: str
    from_role: str = ""
    user_name: str = ""
    image_base64: Optional[str] = None
    received_at: float = field(default_factory=time.time)


# ──────────────────────────────────────────────
#  Edge message routing
# ──────────────────────────────────────────────
async def on_edge_message(data):
    """
    Handles every message the edge hub forwards to this orchestrator.

    Per the hub's static routing table (delta-edge's routing.rs), the
    orchestrator only ever receives two text message types:
      - "user_input"  from the Client role (Unity frontend)
      - "stt_result"  from the Stt role (transcribed speech)
    Both carry text to run inference on; envelope shape is
    {"type": ..., "payload": {...}, "from": "<sender role>"}.
    """
    if isinstance(data, (bytes, bytearray)):
        # the hub only routes binary Tts->Client and Client->Stt — orchestrator never gets binary
        logger.warning("Received unexpected raw bytes from edge")
        return

    msg_type = data.get("type", "")
    payload = data.get("payload", {}) or {}

    if msg_type in ("user_input", "stt_result"):
        job = InferenceJob(
            text=payload.get("text", ""),
            request_id=payload.get("request_id") or str(uuid.uuid4()),
            from_role=data.get("from", ""),
            user_name=payload.get("user_name", ""),
            image_base64=payload.get("image_base64"),
        )
        await orchestrator.submit(job)
        return

    logger.warning(f"Unhandled edge message type: {msg_type!r} from {data.get('from')!r}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Server] Starting Reminh Orchestrator...")

    await edge.connect()
    edge.start_listener(on_edge_message)
    orchestrator.start()
    print("[Server] Servers started.")

    await orchestrator.check_downstream_servers()

    print("[Server] Ready.")
    yield

    print("[Server] Shutting down...")
    await orchestrator.stop()
    await edge.close()

app = FastAPI(lifespan=lifespan)

# ──────────────────────────────────────────────
#  Routes: HTTP (No edge route)
# ──────────────────────────────────────────────
@app.post("/discord/chat")
async def discord_chat_http(request: DiscordBotTextRequest):
    raise NotImplementedError("TODO: 기존 handle_discord_text_inference 로직 이식")


@app.get("/health")
async def health():
    return {"status": "ok", "queue_size": orchestrator._queue.qsize()}


if __name__ == "__main__":
    uvicorn.run(app, host=config.server.host, port=config.server.port)
