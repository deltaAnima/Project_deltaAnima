"""
delta-edge test harness

Simulates service roles to verify edge routing without real STT/TTS/VLM.
Run the edge server first, then:

    python test_edge.py --role client     # pretend to be Unity
    python test_edge.py --role stt        # pretend to be STT and send text
    python test_edge.py --role all        # spawn all roles in one process

Requires: pip install websockets
"""

import argparse
import asyncio
import json
import websockets


EDGE_URL = "ws://127.0.0.1:8080"


async def connect_and_register(role: str):
    ws = await websockets.connect(EDGE_URL, max_size=None)
    await ws.send(json.dumps({"type": "register", "role": role}))
    ack = json.loads(await ws.recv())
    print(f"[{role}] Registered: {ack}")
    return ws


async def listen(ws, role: str):
    """Print everything this role receives from the edge."""
    try:
        async for msg in ws:
            if isinstance(msg, str):
                data = json.loads(msg)
                print(f"[{role}] << {json.dumps(data, ensure_ascii=False, indent=2)}")
            else:
                print(f"[{role}] << binary ({len(msg)} bytes)")
    except websockets.ConnectionClosed:
        print(f"[{role}] Connection closed")


async def run_client():
    """Simulate Unity: connect as client, listen for results."""
    ws = await connect_and_register("client")
    print("[client] Listening for inference results and TTS audio...")
    print("[client] (In another terminal, run with --role stt to trigger the pipeline)\n")
    await listen(ws, "client")


async def run_stt():
    """Simulate STT: connect and send a transcription result."""
    ws = await connect_and_register("stt")

    # Give other services time to connect
    await asyncio.sleep(1)

    text = input("[stt] Enter text to send (or press Enter for default): ").strip()
    if not text:
        text = "Hello, how are you today?"

    msg = {
        "type": "stt_result",
        "payload": {"text": text},
    }
    await ws.send(json.dumps(msg))
    print(f"[stt] >> Sent stt_result: {text}")

    # Listen for any responses
    await listen(ws, "stt")


async def run_mock_orchestrator():
    """
    Simulate orchestrator: receive stt_result, respond with inference_result.
    Use this if the real orchestrator isn't running.
    """
    ws = await connect_and_register("orchestrator")
    print("[orchestrator] Waiting for messages...\n")

    async for msg in ws:
        if isinstance(msg, bytes):
            continue
        data = json.loads(msg)
        print(f"[orchestrator] << {json.dumps(data, ensure_ascii=False, indent=2)}")

        msg_type = data.get("type", "")
        payload = data.get("payload", {})

        if msg_type == "stt_result":
            # Fake inference: echo the text back
            text = payload.get("text", "")
            print(f"[orchestrator] Running fake inference on: {text}")
            await asyncio.sleep(0.5)  # simulate processing

            # Send inference result (routes to client via edge)
            result = {
                "type": "inference_result",
                "payload": {
                    "text": f"[mock response] You said: {text}",
                    "expre_result": {"valence": 0.5, "arousal": 0.3, "dominance": 0.4},
                },
            }
            await ws.send(json.dumps(result))
            print(f"[orchestrator] >> Sent inference_result")

            # Fire TTS request (routes to TTS via edge)
            tts_req = {
                "type": "tts_request",
                "payload": {"text": f"[mock response] You said: {text}"},
            }
            await ws.send(json.dumps(tts_req))
            print(f"[orchestrator] >> Sent tts_request (fire-and-forget)")


async def run_mock_tts():
    """
    Simulate TTS: receive tts_request, send back fake audio chunks.
    Use this if the real TTS server isn't running.
    """
    ws = await connect_and_register("tts")
    print("[tts] Waiting for tts_request...\n")

    async for msg in ws:
        if isinstance(msg, bytes):
            continue
        data = json.loads(msg)
        print(f"[tts] << {json.dumps(data, ensure_ascii=False, indent=2)}")

        if data.get("type") == "tts_request":
            text = data.get("payload", {}).get("text", "")
            print(f"[tts] Synthesizing: {text}")

            # Send tts_start metadata (routes to client)
            await ws.send(json.dumps({
                "type": "tts_start",
                "payload": {"sample_rate": 22050, "format": "pcm_s16le"},
            }))

            # Send fake audio chunks (binary, routes to client)
            for i in range(5):
                fake_audio = bytes([0] * 4096)  # silence
                await ws.send(fake_audio)
                print(f"[tts] >> Sent audio chunk {i + 1}/5")
                await asyncio.sleep(0.1)

            # Send tts_done
            await ws.send(json.dumps({
                "type": "tts_done",
                "payload": {"chunks": 5, "seconds": 0.5},
            }))
            print("[tts] >> Done")


async def run_all():
    """Spawn all mock roles in one process for quick testing."""
    print("=" * 50)
    print("  Starting all mock roles")
    print("  Edge server must be running on ws://127.0.0.1:8080")
    print("=" * 50)
    print()

    # Start listeners
    tasks = [
        asyncio.create_task(run_client()),
        asyncio.create_task(run_mock_orchestrator()),
        asyncio.create_task(run_mock_tts()),
    ]

    # Wait a moment, then trigger with a fake STT result
    await asyncio.sleep(2)

    stt_ws = await connect_and_register("stt")
    msg = {
        "type": "stt_result",
        "payload": {"text": "Hello, testing the full pipeline!"},
    }
    await stt_ws.send(json.dumps(msg))
    print("\n[test] >> Sent stt_result to kick off the pipeline\n")

    # Let it run for a bit
    await asyncio.sleep(5)

    print("\n" + "=" * 50)
    print("  Test complete! Check the logs above.")
    print("=" * 50)

    for t in tasks:
        t.cancel()


def main():
    p = argparse.ArgumentParser(description="delta-edge test harness")
    p.add_argument(
        "--role",
        choices=["client", "stt", "orchestrator", "tts", "all"],
        default="all",
        help="Which role to simulate",
    )
    p.add_argument("--url", default=EDGE_URL, help="Edge server URL")
    args = p.parse_args()

    global EDGE_URL
    EDGE_URL = args.url

    runners = {
        "client": run_client,
        "stt": run_stt,
        "orchestrator": run_mock_orchestrator,
        "tts": run_mock_tts,
        "all": run_all,
    }

    try:
        asyncio.run(runners[args.role]())
    except KeyboardInterrupt:
        print("\n[test] Stopped.")


if __name__ == "__main__":
    main()
