import base64
import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from fastapi import WebSocket
import websockets
from dotenv import load_dotenv

from MainServerHelper.Pydantic_frame import (
    DiscordBotTextRequest,
    ReminhInferenceRequest,
    STTRequest,
    TTSRequest,
    MainAiRequest,
    ReloadYamlRequest,
    DiscordBotTextResponse,
)
from VL.VisionLangHandler import VisionLangHandler

# Reminh
from Persona.Reminh import Reminh


@dataclass
class TTSSettingContainer:
    ws_url: str = "ws://127.0.0.1:8765/tts"
    language: str = "Japanese"
    speaker: str = "Ono_Anna"
    instruct: str = ""
    audio_format: str = "pcm_s16le"
    emit_every_frames: int = 8
    decode_window_frames: int = 80
    max_decode_window_frames: int = 96
    overlap_samples: int = 384
    first_emit_frames: int = 4

    def to_request(self, text: str) -> dict:
        return {
            "text": text,
            "language": self.language,
            "speaker": self.speaker,
            "instruct": self.instruct,
            "format": self.audio_format,
            "emit_every_frames": self.emit_every_frames,
            "decode_window_frames": self.decode_window_frames,
            "max_decode_window_frames": self.max_decode_window_frames,
            "overlap_samples": self.overlap_samples,
            "first_emit_frames": self.first_emit_frames,
        }


@dataclass
class VLSettingContainer:
    api_url: str = "http://localhost:11434/api/generate"
    model_name: str = ""
    timeout: float = 60.0


class ReminhLogicHandler:
    def __init__(self, vl_handler: VisionLangHandler):
        load_dotenv()
        self.tts_setting, self.vl_setting = self._build_settings()
        self.vl_handler: VisionLangHandler = vl_handler

    # ──────────────────────────────────────────────
    #  Main inference pipeline
    # ──────────────────────────────────────────────

    async def handle_main_inference(
        self, Remi: Reminh, websocket: WebSocket, request: ReminhInferenceRequest
    ):
        print(f"[LogicHandler] Main Inference Start: {request.input_type}")
        try:
            current_prompt = request.text or ""
            print(f"[LogicHandler] Starting VLM inference: {current_prompt}")

            # 1. VAD inference
            system_prompt = Remi.get_VAD_prompt(current_prompt)
            vlm_response_container = self.vl_handler.inference(
                user_prompt=f"User input: {current_prompt}",
                system_prompt=system_prompt,
                image_path=request.image_base64,
            )

            if not (vlm_response_container and vlm_response_container[0]):
                print("[LogicHandler] Warning: No text generated from VLM. (VAD)")
                await websocket.send_json({"error": "VLM Generation Failed (VAD)"})
                return

            print("[LogicHandler] VLM inference done! (VAD)")
            vlm_response_text = vlm_response_container[1]
            print(f"DEBUG VLM RESULT: {vlm_response_container}")

            # 2. Reminh inference
            system_prompt = Remi.get_Reminh_prompt(vlm_response_text, current_prompt)
            current_captured_emotion = Remi.get_Reminh_last_emotion().copy()

            vlm_response_container = self.vl_handler.inference(
                user_prompt=current_prompt,
                system_prompt=system_prompt,
                image_path=request.image_base64,
            )

            if not (vlm_response_container and vlm_response_container[0]):
                print("[LogicHandler] Warning: No text generated from VLM. (Reminh)")
                await websocket.send_json({"error": "VLM Generation Failed (Reminh)"})
                return

            print("[LogicHandler] VLM inference done! (Reminh)")
            vlm_response_text = vlm_response_container[1]
            print(f"DEBUG VLM RESULT: {vlm_response_container}")

            # 3. Send text + emotion immediately
            await websocket.send_json({
                "type": "inference_result",
                "text": vlm_response_text,
                "expre_result": current_captured_emotion,
            })

            # 4. Stream TTS
            if vlm_response_text:
                await self._stream_tts(websocket, vlm_response_text)

        except Exception as e:
            print(f"[LogicHandler] Inference Error: {e}")
            await websocket.send_json(
                {"error": f"Inference processing failed: {str(e)}"}
            )

    # ──────────────────────────────────────────────
    #  Main inference pipeline (edge version)
    # ──────────────────────────────────────────────

    async def handle_main_inference_edge(
        self, Remi: Reminh, edge, request: ReminhInferenceRequest
    ):
        """Edge version: sends results via EdgeClient, fire-and-forget TTS."""
        print(f"[LogicHandler] Main Inference Start (edge): {request.input_type}")
        try:
            current_prompt = request.text or ""
            print(f"[LogicHandler] Starting VLM inference: {current_prompt}")

            # 1. VAD inference
            system_prompt = Remi.get_VAD_prompt()
            vlm_response_container = self.vl_handler.inference(
                user_prompt=f"User input: {current_prompt}",
                system_prompt=system_prompt,
                image_path=request.image_base64,
            )

            if not (vlm_response_container and vlm_response_container[0]):
                print("[LogicHandler] Warning: No text generated from VLM. (VAD)")
                await edge.send_routed("error", {"message": "VLM Generation Failed (VAD)"}, to="client")
                return

            print("[LogicHandler] VLM inference done! (VAD)")
            vlm_response_text = vlm_response_container[1]

            # 2. Reminh inference
            system_prompt = Remi.get_Reminh_prompt(vlm_response_text, current_prompt)
            current_captured_emotion = Remi.get_Reminh_last_emotion().copy()

            vlm_response_container = self.vl_handler.inference(
                user_prompt=current_prompt,
                system_prompt=system_prompt,
                image_path=request.image_base64,
            )

            if not (vlm_response_container and vlm_response_container[0]):
                print("[LogicHandler] Warning: No text generated from VLM. (Reminh)")
                await edge.send_routed("error", {"message": "VLM Generation Failed (Reminh)"}, to="client")
                return

            print("[LogicHandler] VLM inference done! (Reminh)")
            vlm_response_text = vlm_response_container[1]

            # 3. Send text + emotion → edge → client
            await edge.send_routed("inference_result", {
                "text": vlm_response_text,
                "expre_result": current_captured_emotion,
            })

            # 4. Fire-and-forget TTS → edge → TTS server
            if vlm_response_text:
                await edge.send_routed("tts_request", {
                    **self.tts_setting.to_request(vlm_response_text),
                })

        except Exception as e:
            print(f"[LogicHandler] Inference Error (edge): {e}")
            try:
                await edge.send_routed("error", {"message": str(e)}, to="client")
            except Exception:
                pass

    # ──────────────────────────────────────────────
    #  TTS streaming (WS → WS relay, legacy)
    # ──────────────────────────────────────────────

    async def _stream_tts(self, websocket: WebSocket, text: str):
        try:
            async with websockets.connect(
                uri=self.tts_setting.ws_url,
                max_size=None,
            ) as tts_ws:
                # Send TTS request
                await tts_ws.send(json.dumps(
                    self.tts_setting.to_request(text)
                ))

                # Relay loop
                async for msg in tts_ws:
                    if isinstance(msg, str):
                        meta = json.loads(msg)
                        if meta["type"] == "start":
                            await websocket.send_json({
                                "type": "tts_start",
                                "sample_rate": meta["sample_rate"],
                                "format": meta["format"],
                            })
                        else:
                            # done
                            await websocket.send_json({"type": "tts_done"})
                            break
                    else:
                        # audio chunk → relay to frontend
                        await websocket.send_bytes(msg)

        except Exception as e:
            print(f"[LogicHandler] TTS Streaming Error: {e}")
            await websocket.send_json({"error": f"TTS failed: {str(e)}"})

    # ──────────────────────────────────────────────
    #  Test handlers
    # ──────────────────────────────────────────────

    async def handle_stt_test(self, websocket: WebSocket, request: STTRequest):
        pass

    async def handle_tts_test(self, websocket: WebSocket, request: TTSRequest):
        pass

    # ──────────────────────────────────────────────
    #  YAML reload
    # ──────────────────────────────────────────────

    async def handle_reload_yaml(
        self, Remi: Reminh, websocket: WebSocket, request: ReloadYamlRequest
    ):
        print("[System] YAML reload request received")
        try:
            success: bool = Remi.reload_physics_config()
            if success:
                print("[System] deltaEGO YAML update successful")
                await websocket.send_json({
                    "response_type": "system_notice",
                    "status": "success",
                    "message": "Reminh's emotion physics settings (YAML) have been successfully updated.",
                })
            else:
                print("[System] deltaEGO YAML update failed")
                await websocket.send_json({
                    "response_type": "system_notice",
                    "status": "error",
                    "message": "Failed to update YAML. (Internal C++ Error)",
                })
        except Exception as e:
            print(f"[Fatal Error] Exception occurred during YAML reload: {e}")
            await websocket.send_json(
                {"response_type": "system_notice", "status": "error", "message": str(e)}
            )

    # ──────────────────────────────────────────────
    #  Discord (HTTP, no TTS)
    # ──────────────────────────────────────────────

    async def handle_discord_text_inference(
        self, Remi: Reminh, request: DiscordBotTextRequest
    ):
        print(f"[LogicHandler] Discord isolated inference for: {request.user_name}")
        try:
            current_prompt = request.message_content
            target_image = request.attachments[0] if request.attachments else None

            # VAD
            vlm_res_vad = self.vl_handler.inference(
                user_prompt=f"User: {current_prompt}",
                system_prompt=Remi.get_VAD_prompt(),
                image_path=target_image,
            )
            vlm_vad_raw = vlm_res_vad[1] if vlm_res_vad and vlm_res_vad[0] else "{}"

            # Reminh
            system_prompt_reminh = Remi.get_Reminh_prompt(
                VAD_result_raw=vlm_vad_raw,
                user_input=current_prompt,
                source="discord_txt",
                user_name=request.user_name,
            )
            vlm_res_reminh = self.vl_handler.inference(
                user_prompt=current_prompt,
                system_prompt=system_prompt_reminh,
                image_path=target_image,
            )

            Remi.set_Reminh_memory(
                User_Name=request.user_name, AI_output=vlm_res_reminh[1]
            )
            last_emotion = Remi.get_Reminh_last_emotion()

            if vlm_res_reminh and vlm_res_reminh[0]:
                return DiscordBotTextResponse(
                    status="success",
                    output_text=vlm_res_reminh[1],
                    emotion_tag=last_emotion.get("emotion_term", "calm"),
                )
            else:
                return DiscordBotTextResponse(
                    status="error", output_text="VLM inference failed"
                )
        except Exception as e:
            print(f"[Critical Discord Error] {e}")
            return DiscordBotTextResponse(status="error", output_text=str(e))

    # ──────────────────────────────────────────────
    #  Settings builder
    # ──────────────────────────────────────────────

    @staticmethod
    def _build_settings() -> tuple[TTSSettingContainer, VLSettingContainer]:
        tts = TTSSettingContainer(
            ws_url=os.environ.get("TTS_SERVER_WS_URL", "ws://127.0.0.1:8765/tts"),
            language=os.environ.get("TTS_LANGUAGE", "Japanese"),
            speaker=os.environ.get("TTS_SPEAKER", "reminh"),
            instruct=os.environ.get("TTS_INSTRUCT", ""),
            audio_format=os.environ.get("TTS_FORMAT", "pcm_s16le"),
            emit_every_frames=int(os.environ.get("TTS_EMIT_EVERY_FRAMES", "8")),
            decode_window_frames=int(os.environ.get("TTS_DECODE_WINDOW_FRAMES", "80")),
            max_decode_window_frames=int(os.environ.get("TTS_MAX_DECODE_WINDOW_FRAMES", "96")),
            overlap_samples=int(os.environ.get("TTS_OVERLAP_SAMPLES", "384")),
            first_emit_frames=int(os.environ.get("TTS_FIRST_EMIT_FRAMES", "4")),
        )
        vl = VLSettingContainer(
            api_url=os.environ.get("VLM_API_URL", "http://localhost:11434/api/generate"),
            model_name=os.environ.get("VLM_MODEL_NAME", ""),
            timeout=float(os.environ.get("VLM_TIMEOUT", "60.0")),
        )
        print(f"[Settings] TTS: {tts.ws_url} | Speaker: {tts.speaker}")
        print(f"[Settings] VLM: {vl.api_url} | Model: {vl.model_name}")
        return tts, vl
