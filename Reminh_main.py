"""
external libs
"""

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from contextlib import asynccontextmanager
from pydantic import ValidationError, TypeAdapter
from dotenv import load_dotenv
from typing import Union
import json
import asyncio
import base64
import time
import os

load_dotenv()

"""
    custom libs
"""
# pydantic classes
from MainServerHelper.Pydantic_frame import *

# setting up server
from setup_servers import *
from VL.VisionLangHandler import VisionLangHandler

# websocket logics
from LogicHandler import ReminhLogicHandler

# Reminh
from Persona.Reminh import Reminh

# edge_client
from edge_client import EdgeClient

print("Welcome!!!")
print("Waking up Reminh...........")

"""
    # 1. setting up server
        * VL model server (5090)
        * STT server (3090)
        * TTS server (3090)
"""

print("Step 1. initializing isolated model servers..........")
setup_tmux()
print("servers are started!")

"""
   # 2. Initializing Handler
        * VisionLangHandler
        * TTS_Handler
"""
print("Step 2.Starting model handlers........")
VL_handler: VisionLangHandler = VisionLangHandler(
    temperature=0.8,
    max_token=3000,
    abs_model_path="~/work/deltaAnima/Reminh/models/Huihui-Qwen3-VL-30B-A3B-Instruct-abliterated-Q4_K_M.gguf",
    model_full="Qwen3-VL-30B-A3B-Instruct-abliterated",
)
logic_handler = ReminhLogicHandler(vl_handler=VL_handler)

print("Loading Handlers are done!!!! Starting websocket server................")
# Waking up Reminh
Remi: Reminh = Reminh()

# ──────────────────────────────────────────────
#  3. Edge client
# ──────────────────────────────────────────────
edge = EdgeClient(
    edge_url=os.environ.get("EDGE_SERVER_URL", "ws://0.0.0.0:4524"),
    role="orchestrator",
)
 
 
async def on_edge_message(data):
    """Dispatch messages arriving from the edge server."""
 
    # Binary data (not expected for orchestrator, but handle gracefully)
    if isinstance(data, bytes):
        print(f"[Edge] Unexpected binary data ({len(data)} bytes)")
        return
 
    msg_type = data.get("type", "")
    payload = data.get("payload", {})
    from_role = data.get("from", "")
 
    print(f"[Edge] Received: type={msg_type} from={from_role}")
 
    if msg_type == "stt_result":
        # STT sent transcribed text via edge
        text = payload.get("text", "")
        if not text:
            return
        request = ReminhInferenceRequest(
            request_type="inference",
            input_type="text",
            text=text,
            image_base64=None,
        )
        await logic_handler.handle_main_inference_edge(Remi, edge, request)
 
    elif msg_type == "user_input":
        # Unity client sent text input via edge
        text = payload.get("text", "")
        image = payload.get("image_base64")
        if not text:
            return
        request = ReminhInferenceRequest(
            request_type="inference",
            input_type="image" if image else "text",
            text=text,
            image_base64=image,
        )
        await logic_handler.handle_main_inference_edge(Remi, edge, request)
 
    else:
        print(f"[Edge] Unhandled message type: {msg_type}")
 

# ──────────────────────────────────────────────
#  4. Legacy WS + connection manager (debug)
# ──────────────────────────────────────────────
class ConnectionManager:
    def __init__(self):
        self.active_connection: Optional[WebSocket] = None
 
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connection = websocket
        print("[Orchestrator] Unity Client Connected (legacy WS)")
 
    def disconnect(self):
        self.active_connection = None
        print("[Orchestrator] Client Disconnected (legacy WS)")
 
 
manager = ConnectionManager()
request_adapter = TypeAdapter(
    Union[ReminhInferenceRequest, STTRequest, TTSRequest, MainAiRequest]
)
 
# ──────────────────────────────────────────────
#  5. FastAPI app
# ──────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("[Server] Starting Reminh Orchestrator...")
    try:
        await edge.connect()
        edge.start_listener(on_edge_message)
        print("[Server] Edge connection established!")
    except Exception as e:
        print(f"[Server] Edge connection failed: {e}")
        print("[Server] Running in legacy mode (direct WS only)")
 
    yield
 
    # Shutdown
    print("[Server] Shutting down. Saving all memories...")
    await edge.close()
    try:
        Remi.MemoryHandler.save_db()
    except Exception as e:
        print(f"[Error] Failed to save memories on shutdown: {e}")


app = FastAPI(lifespan=lifespan)


#  ──────────────────────────────────────────────
#  Routes: Legacy WS (keep for local debugging)
# ──────────────────────────────────────────────
@app.websocket("/ws/reminh")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            raw_data = await websocket.receive_json()
            print(f"LOG [legacy]: {raw_data}")
            try:
                request = request_adapter.validate_python(raw_data)
            except ValidationError:
                await websocket.send_json(
                    {"error": "Unknown Request Type or Invalid Format"}
                )
                continue
 
            if isinstance(request, ReminhInferenceRequest):
                await logic_handler.handle_main_inference(Remi, websocket, request)

            elif isinstance(request, STTRequest):
                await logic_handler.handle_stt_test(websocket, request)

            elif isinstance(request, TTSRequest):
                await logic_handler.handle_tts_test(websocket, request)

            elif isinstance(request, MainAiRequest):
                pass

            elif isinstance(request, ReloadYamlRequest):
                await logic_handler.handle_reload_yaml(Remi, websocket, request)
 
    except WebSocketDisconnect:
        print("[Orchestrator] Client disconnected (legacy)")
        manager.disconnect()
    except Exception as e:
        print(f"[Fatal Error] {e}")
    finally:
        manager.disconnect()
 
 
# ──────────────────────────────────────────────
#  Routes: Legacy STT callback (keep for debug)
# ──────────────────────────────────────────────
@app.post("/stt/callback")
async def stt_callback_handler(request: STTCallBackReq) -> dict[str, str]:
    print(f"[Orchestrator] Text received from STT (legacy callback): {request.text}")
    temp_request = ReminhInferenceRequest(
        request_type="inference",
        input_type="text",
        text=request.text,
        image_base64=None,
    )
    if manager.active_connection:
        asyncio.create_task(
            logic_handler.handle_main_inference(
                Remi, websocket=manager.active_connection, request=temp_request
            )
        )
    return {"status": "success"}
 
 
# ──────────────────────────────────────────────
#  Routes: HTTP (these stay as-is, no edge needed)
# ──────────────────────────────────────────────
@app.post("/discord/chat")
async def discord_chat_http(request: DiscordBotTextRequest):
    response = await logic_handler.handle_discord_text_inference(Remi, request)
    return response.model_dump()
 
 
@app.post("/admin/reload")
async def reload_prompt():
    Remi.PromptModule.reload()
    return {"status": "success", "message": "Prompt reloaded"}
