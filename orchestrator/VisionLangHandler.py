#
# This class will Handle the Vision Language model
# Backend: SGLang (OpenAI-compatible API)
#

from typing import Union, Optional, List, Dict, Any
from pathlib import Path
import requests
import base64
import mimetypes
import time

class VisionLangHandler:
    """
        This Class will handle our VL(Vision&Language) model in 5090.
    """
    def __init__(
            self,
            address         : str = "0.0.0.0",
            port            : str = "30000",
            alias           : str = "Qwen3-VL",
            temperature     : float = 0.7,
            max_token       : int = 5000,
            top_p           : float = 0.9,
            model_full      : str = "",
            enable_thinking : Optional[bool] = None,
            timeout         : float = 60
            ):
        """
        Constructor of the VisonLangHandler. This is purely a remote HTTP
        client for an SGLang server (OpenAI-compatible API) running on its
        own machine — it never loads or manages the model itself, so it has
        no notion of a local GPU device or model file path.

        Args name         | <type>       | default value      | explained
        -------------------------------------------------------------------------------------------------------
        1. address        : <str>          default = "0.0.0.0"  -> server(SGLang) ip address
        2. port           : <str>          default = "30000"    -> port number of the server
        3. alias          : <str>          default = "Qwen3-VL" -> served model name (SGLang --served-model-name)
        4. temperature    : <float>        default = 0.7        -> temperature(creativity) of the AI
        5. max_token      : <int>          default = 5000       -> max_token for thinking and output
        6. top_p          : <float>        default = 0.9        -> it can control PDF in ANN
        7. model_full     : <str>          default = ""         -> full name of the model (SGLang model_path)
        8. enable_thinking: <bool|None>    default = None       -> chat_template_kwargs.enable_thinking. None = server default
        9. timeout        : <float>        default = 60         -> seconds to wait for one completion
        """
        self.address = address
        self.port = port
        self.alias = alias
        self.temperature = temperature
        self.max_token = max_token
        self.top_p = top_p
        self.model_full = model_full
        self.enable_thinking = enable_thinking
        self.timeout = timeout


    def check_server(self) -> List:
        """
            Function check_server check the server is running and model is properly deployed on the server

            Args: N/A

            Return: list[bool, str]
                if bool = true
                    -> It will return [true, ""]
                else
                    -> It will return [false, "reason"]
        """
        url = f"http://{self.address}:{self.port}/get_model_info"
        try:
            # this is a remote call over the LAN, so a short timeout is enough to tell "down" from "slow"
            response = requests.get(url, timeout = 5)

            if response.status_code == 200:
                info = response.json()
                # SGLang reports both the served name (--served-model-name) and the
                # HF repo / local path the weights came from — match either one
                served_name = info.get("served_model_name", "") or ""
                model_path = info.get("model_path", "") or ""
                if (self.alias and self.alias in served_name) or (self.model_full and self.model_full in model_path):
                    # inference() sends images, so a text-only model is a misconfiguration
                    if info.get("has_image_understanding") is False:
                        return [False, f"Model {served_name} has no image understanding"]
                    return [True, ""]
                # model name mismatch
                else:
                    return [False, f"Model not found. Loaded: {served_name} ({model_path})"]
            # server is not on or server error
            else:
                return [False, f"Server Error: {response.status_code}"]
        # Error
        except Exception as e:
            return [False, f"Connection Failed: {str(e)}"]

    def inference(
            self,
            user_prompt     : str,
            system_prompt   : str,
            image_path      : str | None = None,
            image_base64    : str | None = None,
            request_id      : str | None = None,
            history         : list[dict[str, Any]] | None = None
            ) -> list[Any]:
        """
        Gets text or/and image and send to VL to get response

        Args name           | <type>            | default value  | explained
        ---------------------------------------------------------------------------
        1. user_prompt      : <str> *required   | default = N/A  | user input. it can be chat or STT result
        2. system_prompt    : <str> *required   | default = N/A  | system prompt. Crucial for Reminh's persona and memory
        3. image_path       : <str> *optional   | default = None | abs path of the image. It will be used when user shows Reminh image in the computer
        4. image_base64     : <str> *optional   | default = None | encoded image by base64. It is for Reminh's vision
        5. request_id       : <str> *optional   | default = None | sent as SGLang `rid`, so the server-side id matches ours
        6. history          : <list> *optional  | default = None | earlier user/assistant messages, oldest first, replayed before user_prompt

        Return: list[bool(success or not), str(Reminh's response or error reason), dict(meta)]
            meta = {
                "id"                : SGLang request id (== request_id when given)
                "model"             : served model name
                "finish_reason"     : "stop" | "length" | ...
                "reasoning_content" : thinking text, or None
                "usage"             : {prompt_tokens, completion_tokens, reasoning_tokens, cached_tokens}
                "latency_ms"        : round-trip time of the HTTP call
            }
            On a connection/server error only "id" and "latency_ms" are filled.
        """
        url = f"http://{self.address}:{self.port}/v1/chat/completions"
        meta: dict[str, Any] = {"id": request_id, "latency_ms": None}
        # container for user prompt
        content_list: list[dict[str, Any]] = [{"type": "text", "text": user_prompt}]

        # image with abs path
        if image_path:
            try:
                mime_type, _ = mimetypes.guess_type(image_path)
                mime_type = mime_type or "image/jpeg"
                # read img with abs path and encode img with base64
                with open(image_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                # add it to content_list
                content_list.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{encoded_string}"}
                })
            # Base64 encoding error
            except Exception as e:
                return [False, f"Image Error: {str(e)}", meta]
        # image with Reminh's vision
        elif image_base64:
            # if the format is wrong, this will fix it
            if not image_base64.startswith("data:"):
                image_base64 = f"data:image/jpeg;base64,{image_base64}"

            content_list.append({
                "type": "image_url",
                "image_url": {"url": image_base64}
            })

        # build request
        # earlier turns sit between the system prompt and the current input
        messages = [
            {"role": "system", "content": system_prompt},
            *(history or []),
            {"role": "user", "content": content_list}
        ]
        body: dict[str, Any] = {
            "model": self.alias,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_token,
            "top_p": self.top_p,
        }
        # SGLang-specific extras
        if request_id:
            body["rid"] = request_id
        if self.enable_thinking is not None:
            body["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}

        # send request
        started = time.perf_counter()
        try:
            response = requests.post(url, json=body, timeout=self.timeout)
        # Connection error
        except Exception as e:
            meta["latency_ms"] = int((time.perf_counter() - started) * 1000)
            return [False, f"Connection Failed: {str(e)}", meta]
        meta["latency_ms"] = int((time.perf_counter() - started) * 1000)

        # somehow Failed
        if response.status_code != 200:
            return [False, f"Server Error: {response.status_code} {response.text[:300]}", meta]

        try:
            data = response.json()
            choice = data["choices"][0]
            message = choice["message"]
        except Exception as e:
            return [False, f"Bad Response: {str(e)}", meta]

        usage = data.get("usage") or {}
        # cached_tokens is only filled when SGLang runs with --enable-cache-report
        prompt_details = usage.get("prompt_tokens_details") or {}
        meta.update({
            "id": data.get("id", request_id),
            "model": data.get("model"),
            "finish_reason": choice.get("finish_reason"),
            "reasoning_content": message.get("reasoning_content"),
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "reasoning_tokens": usage.get("reasoning_tokens"),
                "cached_tokens": prompt_details.get("cached_tokens"),
            },
        })

        content = (message.get("content") or "").strip()
        # with thinking on, the whole max_tokens budget can be spent on
        # reasoning_content, leaving no actual answer to speak
        if not content:
            return [False, f"Empty Response (finish_reason={meta['finish_reason']})", meta]

        # success
        return [True, content, meta]

    def __str__(self) -> str:
        """
        when print(class) or class.__str__(), this will return its data in a right format

        Args: N/A

        Return: str(info of this class)
        """
        return f"""
        Class name: VisionLangHandler

        Handling server info:
            backend : SGLang
            address : {self.address}
            port    : {self.port}
            url     : http://{self.address}:{self.port}/v1/chat/completions
            model   : {self.alias}

        Model info:
            Model Alias         : {self.alias}
            Model full name     : {self.model_full}
            max_tokens          : {self.max_token}
            top_p               : {self.top_p}
            temperature         : {self.temperature}
            enable_thinking     : {self.enable_thinking}
            timeout             : {self.timeout}
        """
