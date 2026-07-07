from pydantic import BaseModel, Field
from typing import Dict, Optional, Literal


"""
    BaseModel of every request++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""


class BaseOrchestratorRequest(BaseModel):
    """
    This is BaseModel of every request.

    requset_type    str      inference, STT_test, TTS_test, VL_test
    """

    pass  # NOTICE: I did this becuz of error with Liskov Substitution Principle


"""
    ============================================================================================
"""

"""
    inference request and response++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""


class ReminhInferenceRequest(BaseOrchestratorRequest):
    """
    This Pydantic container is reqesting inference Reminh's reaction

    Specification:
        Name            | input types   | contents                      |
        ----------------------------------------------------------------|
        input_type      | str           | "text", "image", "audio",     |
                        |               | "text-image", "audio_image",  |
                        |               | "text-audio-image"            |
        ----------------------------------------------------------------|
        text            | str           | general text                  |
        ----------------------------------------------------------------|
        image_base64    | str           | encoded with base64           |
        ----------------------------------------------------------------|
        audio_bytes     | str           | Serialized audio data         |
    """

    # TODO: add image_path
    request_type: Literal["inference"]

    input_type: str
    text: Optional[str] = ""
    image_base64: Optional[str] = None
    audio_bytes: Optional[str] = None


class ReminhInferenceResponse(BaseModel):
    """
    This Pydantic container is providing the result of Reminh's inference.

    Specification:
        Name            | Output Types  | Contents                      |
        ----------------------------------------------------------------|
        status          | str           | "success", "error",           |
                        |               | "processing"                  |
        ----------------------------------------------------------------|
        output_text     | str           | generated response text (VLM) |
        ----------------------------------------------------------------|
        audio_bytes_resp| str           | synthesized audio (TTS)       |
        ----------------------------------------------------------------|
        emotion_tag     | str           | emotion label from deltaEGO   |
        ----------------------------------------------------------------|
        emotion_score   | float         | intensity of the emotion      |
    """

    status: str
    output_text: Optional[str] = None
    audio_bytes_resp: Optional[str] = None
    # TODO: make deltaEGO can send simple emotion as return *NOT YET*
    emotion_tag: Optional[str] = None
    emotion_score: float = 0.0


"""
    ===========================================================================================
"""


"""
    For STT test+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""


class STTRequest(BaseOrchestratorRequest):
    """
    This Pydantic container is for testing Live Whisper (Streaming STT).

    Specification:
        Name            | Input Types   | Contents                               |
        -------------------------------------------------------------------------|
        stream_id       | str           | unique ID for current audio stream     |
        audio_chunk     | str           | base64 encoded short audio fragment    |
        is_final        | bool          | true if this is the end of the stream  |
        sample_rate     | int           | audio sampling rate (default: 16000)   |
        -------------------------------------------------------------------------|
    """

    request_type: Literal["STT_test"]

    stream_id: str
    audio_chunk: str
    is_final: bool = False
    sample_rate: int = 16000


class STTResponse(BaseModel):
    """
    This Pydantic container provides real-time transcription results.

    Specification:
        Name            | Output Types  | Contents                               |
        -------------------------------------------------------------------------|
        status          | str           | "partial", "final", "error"            |
        text            | str           | recognized text from current fragment  |
        latency         | float         | inference time for this chunk (ms)     |
        -------------------------------------------------------------------------|
    """

    status: str  # "partial" (mid result), "final" (final result)
    text: Optional[str] = None
    latency: float = 0.0


"""
    ============================================================================================
"""


"""
    For TTS test++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""


class TTSRequest(BaseOrchestratorRequest):
    """
    This Pydantic container is for testing Text-to-Speech synthesis.

    Specification:
        Name            | Input Types   | Contents                      |
        ----------------------------------------------------------------|
        text            | str           | text to synthesize            |
        voice_id        | str           | specific voice profile ID     |
        speed           | float         | speech speed (0.5 ~ 2.0)      |
    """

    request_type: Literal["TTS_test"]

    text: str
    voice_id: Optional[str] = "Reminh_v1"
    speed: Optional[float] = 1.0


class TTSResponse(BaseModel):
    """
    Specification:
        Name            | Output Types  | Contents                      |
        ----------------------------------------------------------------|
        status          | str           | "success", "error"            |
        audio_bytes_resp| str           | synthesized audio (base64)    |
        duration        | float         | length of the audio in seconds|
    """

    status: str
    audio_bytes_resp: Optional[str] = None
    duration: float = 0.0


"""
    ============================================================================================
"""


"""
    For main Ai test++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""


class MainAiRequest(BaseOrchestratorRequest):
    """
    This Pydantic container is for testing the VLM (Qwen3-VL) directly.

    Specification:
        Name            | Input Types   | Contents                      |
        ----------------------------------------------------------------|
        prompt          | str           | main instruction for AI       |
        image_base64    | str           | target image for analysis     |
        max_tokens      | int           | limit of response length      |
    """

    request_type: Literal["VL_test"]

    prompt: str
    image_base64: Optional[str] = None


class MainAiResponse(BaseModel):
    """
    Specification:
        Name            | Output Types  | Contents                      |
        ----------------------------------------------------------------|
        status          | str           | "success", "error"            |
        generated_text  | str           | raw inference output          |
        inference_time  | float         | time taken (seconds)          |
    """

    status: str
    generated_text: Optional[str] = None
    inference_time: float = 0.0


"""
    ============================================================================================
"""

"""
    For main orchestrator server test+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""

"""
    ============================================================================================
"""


class ReloadYamlRequest(BaseOrchestratorRequest):
    request_type: Literal["reload_yaml"]


"""
    Discord Bot specific request and response +++++++++++++++++++++++++++++++++++++++++++++++++++
"""


class DiscordBotTextRequest(BaseOrchestratorRequest):
    """
    This Pydantic container is for Discord Bot interactions.

    Specification:
        Name            | Input Types   | Contents                      |
        ----------------------------------------------------------------|
        user_id         | str           | unique Discord user ID        |
        user_name       | str           | Discord global name or nick   |
        guild_id        | Optional[str] | ID of the server (if not DM)  |
        channel_id      | str           | ID of the text channel        |
        message_content | str           | raw message text              |
        attachments     | List[str]     | List of base64 encoded images |
        is_dm           | bool          | True if it's a direct message |
        ----------------------------------------------------------------|
    """

    request_type: Literal["discord_chat"]

    user_id: str
    user_name: str
    guild_id: Optional[str] = None
    channel_id: str
    message_content: str
    attachments: Optional[list[str]] = Field(default_factory=list)
    is_dm: bool = False


class DiscordBotTextResponse(BaseModel):
    """
    Response container optimized for Discord output.

    Specification:
        Name            | Output Types  | Contents                      |
        ----------------------------------------------------------------|
        status          | str           | "success", "error"            |
        output_text     | str           | AI generated response         |
        ----------------------------------------------------------------|
    """

    status: str
    output_text: bool | str
    emotion_tag: Optional[str] = None


class STTCallBackReq(BaseModel):
    client_id: str
    text: str
