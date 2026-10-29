"""
external libs
"""
import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from dotenv import load_dotenv
from typing import Optional
import argparse
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
    DiscordBotTextRequest, DiscordBotTextResponse
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
    enable_thinking=config.vl_model.enable_thinking,
    timeout=config.vl_model.timeout,
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
    # stable identity, "<platform>:<platform_user_id>" — keys Fuli memory/session
    user_key: str
    # where the user is talking, e.g. "discord:<guild_id>:<channel_id>" — bounds Fuli retrieval/session
    scope_key: str
    # display name only (prompt, saved memory); may change freely
    user_name: str
    platform: str
    platform_user_id: str
    from_role: str = ""
    image_base64: Optional[str] = None
    # set for HTTP callers (Discord): the result comes back here instead of over the edge
    reply: Optional[asyncio.Future] = None
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
    
    #temp
    print(payload)
    print(payload.get("text"))

    if msg_type in ("user_input", "stt_result"):
        request_id = payload.get("request_id") or str(uuid.uuid4())
        # user_name keys the user's Fuli memory/session, so never run inference without one
        user_name = (payload.get("user_name") or "").strip()
        if not user_name:
            logger.warning(f"Rejected {msg_type!r} without user_name (request_id={request_id})")
            await edge.send_routed("inference_result", {
                "request_id": request_id,
                "status": "error",
                "output_text": None,
                "error": "user_name is required",
                "memory": None,
            })
            return

        platform = (payload.get("platform") or "").strip().lower()
        platform_user_id = str(payload.get("platform_user_id") or "").strip()
        if not platform or not platform_user_id or ":" in platform:
            # old clients only send user_name; keep them working, but they still
            # collide across platforms/same names until they send real ids
            logger.warning(
                f"{msg_type!r} without platform/platform_user_id, falling back to legacy key "
                f"(request_id={request_id}, user_name={user_name!r})"
            )
            platform, platform_user_id = "legacy", user_name

        # edge clients (Unity) may name a room/scene; otherwise one shared scope per platform
        scope_id = str(payload.get("scope_id") or "").strip() or "default"

        job = InferenceJob(
            text=payload.get("text", ""),
            request_id=request_id,
            user_key=f"{platform}:{platform_user_id}",
            scope_key=f"{platform}:{scope_id}",
            user_name=user_name,
            platform=platform,
            platform_user_id=platform_user_id,
            from_role=data.get("from", ""),
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
# Discord jobs share the single inference worker with Unity, so this has to
# cover time spent queued behind other turns, not just one inference.
DISCORD_REPLY_TIMEOUT = 180.0


@app.post("/discord/chat", response_model=DiscordBotTextResponse)
async def discord_chat_http(request: DiscordBotTextRequest) -> DiscordBotTextResponse:
    request_id = str(uuid.uuid4())

    user_name = request.user_name.strip()
    user_id = request.user_id.strip()
    if not user_name or not user_id:
        logger.warning(f"Rejected discord chat without user_name/user_id (request_id={request_id})")
        return DiscordBotTextResponse(request_id=request_id, status="error", error="user_name and user_id are required")

    # a DM is its own scope, so nothing said there can surface in a server channel
    if request.is_dm or not request.guild_id:
        scope_key = f"discord:dm:{request.channel_id}"
    else:
        scope_key = f"discord:{request.guild_id}:{request.channel_id}"

    reply: asyncio.Future = asyncio.get_running_loop().create_future()
    job = InferenceJob(
        text=request.message_content,
        request_id=request_id,
        user_key=f"discord:{user_id}",
        scope_key=scope_key,
        user_name=user_name,
        platform="discord",
        platform_user_id=user_id,
        from_role="discord",
        # the VL handler takes one image per turn
        image_base64=request.attachments[0] if request.attachments else None,
        reply=reply,
    )
    await orchestrator.submit(job)

    try:
        result = await asyncio.wait_for(reply, timeout=DISCORD_REPLY_TIMEOUT)
    except asyncio.TimeoutError:
        # the job stays queued and still runs (and saves to Fuli) later; only this reply is lost
        logger.warning(f"Discord reply timed out (request_id={request_id})")
        return DiscordBotTextResponse(request_id=request_id, status="error", error="inference timed out")
    except Exception as e:
        return DiscordBotTextResponse(request_id=request_id, status="error", error=f"inference failed: {e}")

    emotion_terms = ((result.get("memory") or {}).get("memory") or {}).get("emotion", {}).get("emotion_terms") or []
    return DiscordBotTextResponse(
        request_id=request_id,
        status=result["status"],
        output_text=result["output_text"],
        error=result["error"],
        emotion_tag=emotion_terms[0] if emotion_terms else None,
    )



@app.post("/character/emotion/reload")
async def reload_emotion_params():
    """Re-reads the character's physics_weights.yaml so OCEAN/weights changes apply without a restart."""
    try:
        character.Evernight.reload()
    except Exception as e:
        logger.exception("Emotion param reload failed")
        return {"status": "error", "error": str(e)}
    return {"status": "success", "character": character.name, "OCEAN": character.Evernight.ocean}


@app.post("/character/prompt/reload")
async def reload_prompts():
    """Re-reads the character's prompt.yaml (+ shared VAD prompt) so prompt edits apply without a restart."""
    try:
        character.prompt_handler.reload()
    except Exception as e:
        logger.exception("Prompt reload failed")
        return {"status": "error", "error": str(e)}
    return {"status": "success", "character": character.name}


@app.get("/health")
async def health():
    return {"status": "ok", "queue_size": orchestrator._queue.qsize()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reminh orchestrator")
    parser.add_argument(
        "--tui",
        action="store_true",
        help="run with the terminal dashboard (queue, downstream servers, live log, reload hotkeys); "
             "needs the optional textual package, see requirements-tui.txt",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="with --tui: start with verbose output on (this project's DEBUG records: prompts, Fuli hits and "
             "emotion, VL timings and token usage); press v in the dashboard to switch it",
    )
    args = parser.parse_args()
    if args.verbose and not args.tui:
        parser.error("-v/--verbose only applies together with --tui")

    if args.tui:
        # lazy import: a plain headless run must never depend on textual
        try:
            from tui import run_with_tui
        except ModuleNotFoundError as e:
            if (e.name or "").split(".")[0] != "textual":
                raise
            raise SystemExit("--tui needs the optional 'textual' package: pip install -r requirements-tui.txt")
        # hands uvicorn to the dashboard so both share one event loop; see tui.py
        raise SystemExit(
            run_with_tui(
                app,
                orchestrator=orchestrator,
                edge=edge,
                reload_prompt=reload_prompts,
                reload_emotion=reload_emotion_params,
                verbose=args.verbose,
            )
        )

    uvicorn.run(app, host=config.server.host, port=config.server.port)
