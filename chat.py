import httpx
import json
from typing import List, Optional, Dict, Any, Union

class ImageContent:
    def __init__(self, image_base64: str):
        self.image_base64 = image_base64

    def to_dict(self):
        return {"type": "image", "image_base64": self.image_base64}

class UserMessage:
    def __init__(self, text: str, file_contents: Optional[List[ImageContent]] = None):
        self.text = text
        self.file_contents = file_contents or []

    def to_dict(self):
        msg = {"role": "user", "content": [{"type": "text", "text": self.text}]}
        for item in self.file_contents:
            msg["content"].append(item.to_dict())
        return msg

class LlmChat:
    def __init__(self, api_key: str, session_id: str, system_message: Optional[str] = None):
        self.api_key = api_key
        self.session_id = session_id
        self.system_message = system_message
        self.provider = "gemini"
        self.model = "gemini-pro"
        self.base_url = "https://api.emergentagent.com/v1"

    def with_model(self, provider: str, model: str):
        self.provider = provider
        self.model = model
        return self

    async def send_message(self, message: Union[UserMessage, str]) -> str:
        if isinstance(message, str):
            message = UserMessage(text=message)
        
        payload = {
            "session_id": self.session_id,
            "provider": self.provider,
            "model": self.model,
            "messages": []
        }
        
        if self.system_message:
            payload["messages"].append({"role": "system", "content": self.system_message})
            
        payload["messages"].append(message.to_dict())

        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{self.base_url}/chat",
                headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                json=payload
            )
            
            if resp.status_code != 200:
                raise Exception(f"Emergent API Error {resp.status_code}: {resp.text}")
            
            data = resp.json()
            return data.get("content", "")
