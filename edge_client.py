from typing import Callable, Awaitable, Union
import asyncio
import json
import websockets

class EdgeClient:
    def __init__(self, edge_url: str, role: str) -> None:
        self.edge_url: str = edge_url
        self.role: str = role
        self.ws = None

    async def connect(self):
        self.ws = await websockets.connect(self.edge_url, max_size=None)
        await self.ws.send(json.dumps({
            "type": "register",
            "role": self.role,
        }))
        ack = json.loads(await self.ws.recv())
        print(f"[EdgeClient] Connected to edge, registered as: {ack.get('role')}")
 
    async def send_json(self, data: dict):
        if self.ws:
            await self.ws.send(json.dumps(data))
 
    async def send_bytes(self, data: bytes):
        if self.ws:
            await self.ws.send(data)
 
    async def send_routed(self, msg_type: str, payload: dict, to: str = None):
        msg = {"type": msg_type, "payload": payload}
        if to:
            msg["to"] = to
        await self.send_json(msg)
 
    def start_listener(self, handler: Callable[[Union[dict, bytes]], Awaitable[None]]):
        self._listen_task = asyncio.create_task(self._listen_loop(handler))
 
    async def _listen_loop(self, handler):
        try:
            async for msg in self.ws:
                if isinstance(msg, str):
                    data = json.loads(msg)
                    await handler(data)
                else:
                    await handler(msg)
        except websockets.ConnectionClosed:
            print("[EdgeClient] Edge connection lost")
        except Exception as e:
            print(f"[EdgeClient] Listener error: {e}")
 
    async def close(self):
        if self._listen_task:
            self._listen_task.cancel()
        if self.ws:
            await self.ws.close()
