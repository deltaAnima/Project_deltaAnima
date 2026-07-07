# 
# This class will Handle the Vision Language model
# Currently, only reads text
# TODO -> make it able to send the photos to the model
#

from typing import Union, Optional, List, Dict, Any
from pathlib import Path
import requests
import base64
import mimetypes

class VisionLangHandler:
    """
        This Class will handle our VL(Vision&Language) model in 5090.
    """
    def __init__(
            self,
            address         : str = "0.0.0.0",
            port            : str = "28401",
            alias           : str = "Qwen3-VL",
            temperature     : float = 0.7,
            max_token       : int = 5000,
            top_p           : float = 0.9,
            cuda_name       : str = "cuda:0",
            abs_model_path  : str = "",
            model_full      : str = ""
            ):
        """
        Constructor of the VisonLangHandler.

        Args name         | <type>       | default value      | explained
        -------------------------------------------------------------------------------------------------------
        1. address        : <str>          default = "0.0.0.0"  -> server(llama.cpp) ip address
        2. port           : <str>          default = "28401"    -> port number of the server
        3. alias          : <str>          default = "Qwen3-VL" -> alias(nick name of the model) of the current VL model
        4. temperature    : <float>        default = 0.7        -> temperature(creativity) of the AI
        5. max_token      : <int>          default = 2000       -> max_token for input, thinking and output
        6. top_p          : <float>        default = 0.9        -> it can control PDF in ANN
        7. cuda_name      : <str>          default = "cuda:0"   -> which device(GPU) is using for server
        8. abs_model_path : <str>|<Path>   default = ""         -> absolute path of the model
        9. model_full     : <str>          default = ""         -> full name of the model
        """
        #TODO: add Flash atten(bool), ctx num(int) ,and etc 
        self.address = address
        self.port = port
        self.alias = alias
        self.temperature = temperature
        self.max_token = max_token
        self.top_p = top_p
        self.cuda_name = cuda_name
        self.abs_model_path = abs_model_path
        self.model_full = model_full


    def check_server(self) -> List:
        """
            Function check_server check the server is running and model is properly deployed on the server

            Args: N/A

            Return: Union[bool, str]
                if bool = ture
                    -> It will return [true, ""]
                else
                    -> It #TODO: add Flash atten(bool), ctx num(int) ,and etc will return [false, "reason"]
        """
        url = f"http://{self.address}:{self.port}/v1/models"
        try:
            # check server connection (since it is local based, timeout is set to 5 sec)
            response = requests.get(url, timeout = 5)

            if response.status_code == 200:
                models_data = response.json().get("data", [])
                # check model name (comparing with self.alias and self.model_full)
                loaded_models = [m.get("id") for m in models_data]
                # success
                if (self.alias in loaded_models) or (self.model_full in loaded_models):
                    return [True, ""]
                # model name mismatch
                else:
                    return [False, f"Model not found. Loaded: {loaded_models}"]
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
            image_base64    : str | None = None 
            ) -> list[Union[bool,str]]:
        """
        Gets text or/and image and send to VL to get response

        Args name           | <type>            | default value  | explained
        ---------------------------------------------------------------------------
        1. user_prompt      : <str> *required   | default = N/A  | user input. it can be chat or STT result
        2. system_prompt    : <str> *required   | default = N/A  | system prompt. Crucial for Reminh's persona and memory
        3. image_path       : <str> *optional   | default = None | abs path of the image. It will be used when user shows Reminh image in the computer
        4. image_base64     : <str> *optional   | default = None | encoded image by base64. It is for Reminh's vision

        Return: list[bool(success or not),str(Reminh's response)]
        """
        url = f"http://{self.address}:{self.port}/v1/chat/completions"
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
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}"}
                })
            # Base64 encoding error
            except Exception as e:
                return [False, f"Image Error: {str(e)}"]        
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
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content_list}
        ]
        
        # send request and 
        try:
            response = requests.post(url, json={
                "model": self.alias,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_token
            }, timeout=60)
            
            # success
            if response.status_code == 200:
                return [True, response.json()['choices'][0]['message']['content']]
            # somehow Failed
            else:
                return [False, f"Server Error: {response.status_code}"]
        # Connection error   
        except Exception as e:
            return [False, f"Connection Failed: {str(e)}"]

    def __str__(self) -> str:
        """
        when print(class) or class.__str__(), this will return its data in a right format
        
        Args: N/A

        Return: str(info of this class)
        """
        #TODO: add Flash atten(bool), ctx num(int) ,and etc
        return f"""
        Class name: VisionLangHandler
        
        Handling server info:
            address : {self.address}
            port    : {self.port}
            url     : http://{self.address}:{self.port}/v1/models
            model   : {self.alias}
        
        Model info:
            Model Alias         : {self.alias}
            Model full name     : {self.model_full}
            Model absolute path : {self.abs_model_path}
            max_tokens          : {self.max_token}
            top_p               : {self.top_p}
            temperature         : {self.temperature}

        Using hardware:
            cuda device     : {self.cuda_name}
        """
