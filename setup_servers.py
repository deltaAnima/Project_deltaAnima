from pathlib import Path
import libtmux
import os
import sys

from libtmux.constants import PaneDirection

SESSION_NAME = "Model_Mornitoring"
PROJECT_DIR = Path(__file__).resolve().parent
HOME = Path.home()
NVME_DB = HOME / "mnt/nvme_db"

# Qwen3 TTS
TMUX_TTS = [
    f"cd {PROJECT_DIR}/TTS",
    "export CUDA_VISIBLE_DEVICES=1",
    "conda activate qwen3-tts",
    "python ./websockets/streaming_ws_server.py --port 9880 --model finetuning/output/checkpoint-epoch-2/",
]

TMUX_STT = [
    f"cd {PROJECT_DIR}/STT/",
    "export CUDA_VISIBLE_DEVICES=1",
    "source ./.venv/bin/activate",
    "cd ArgaliaEars_v.1.1_fastapi",
    "taskset -c 28-31 uvicorn STT_api_calls:app --host 0.0.0.0 --port 3090",
]

TMUX_VL = [
    f"cd {NVME_DB}/llama_cpp/llama.cpp/build/",
    "export CUDA_VISIBLE_DEVICES=0",
    f"./bin/llama-server "
    f"-m {NVME_DB}/models/VLM/Gemma4-26B-A4B-Uncensored-HauhauCS-Balanced-Q4_K_M.gguf "
    "--n-gpu-layers -1 "
    "--ctx-size 131072 "
    "--host 0.0.0.0 "
    "--port 28401 "
    "-fa on "
    "-ctk q8_0 "
    "--chat-template-kwargs '{\"enable_thinking\":false}' "
    "-ctv q8_0",
]


def setup_tmux():
    server = libtmux.Server()
    if server.has_session(SESSION_NAME):
        print(f"NOTICE!!!: Session '{SESSION_NAME}' already exists.")
        return

    print(f"Initializing Reminh's nerve system(internal servers): {SESSION_NAME}...")

    session = server.new_session(session_name=SESSION_NAME)
    win = session.windows[0]
    win.rename_window("Monitoring")

    pane_tts = win.panes[0]
    pane_stt = win.split(direction=PaneDirection("RIGHT"))
    pane_vl = pane_stt.split(direction=PaneDirection("RIGHT"))

    def run_commands(pane, cmds, name):
        print(f"Starting {name}...")
        pane.send_keys("source ~/.bashrc")
        for cmd in cmds:
            pane.send_keys(cmd)

    # TTS:win
    run_commands(pane_tts, TMUX_TTS, "TTS (GPU 1)")
    # STT
    #if "TMUX_STT" in globals() and TMUX_STT:
        #run_commands(pane_stt, TMUX_STT, "STT (GPU 1)")
    # VL model
    if "TMUX_VL" in globals() and TMUX_VL:
        run_commands(pane_vl, TMUX_VL, "VL (GPU 0)")

    win.select_layout("tiled")

    print("\nNerve Setup Complete! Attaching...")


if __name__ == "__main__":
    setup_tmux()
