import asyncio
import json
import re
from typing import Any, Dict, List

import httpx

from app.core.config import Settings


class LLMClient:
    def __init__(self, settings: Settings):
        self.settings = settings

    async def generate_json(self, system_prompt: str, user_prompt: str, retries: int = 3) -> Dict[str, Any]:
        for attempt in range(retries):
            try:
                text = await self._generate_with_fallback(system_prompt, user_prompt)
                return self._extract_json(text)
            except Exception:
                if attempt == retries - 1:
                    raise
                await asyncio.sleep(0.7 * (attempt + 1))
        return {}

    async def _generate_with_fallback(self, system_prompt: str, user_prompt: str) -> str:
        if self.settings.groq_api_key:
            try:
                return await self._call_groq(system_prompt, user_prompt)
            except Exception:
                pass

        if self.settings.huggingface_api_key:
            return await self._call_huggingface(system_prompt, user_prompt)

        raise RuntimeError("No LLM provider configured. Set GROQ_API_KEY or HUGGINGFACE_API_KEY.")

    async def _call_groq(self, system_prompt: str, user_prompt: str) -> str:
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.settings.groq_api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.settings.groq_model,
            "temperature": 0.1,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        async with httpx.AsyncClient(timeout=self.settings.llm_timeout_sec) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

    async def _call_huggingface(self, system_prompt: str, user_prompt: str) -> str:
        url = f"https://api-inference.huggingface.co/models/{self.settings.huggingface_model}"
        headers = {
            "Authorization": f"Bearer {self.settings.huggingface_api_key}",
            "Content-Type": "application/json",
        }
        prompt = (
            "You are a precise research assistant. Return valid JSON only.\n\n"
            f"System: {system_prompt}\n\n"
            f"User: {user_prompt}"
        )
        payload: Dict[str, Any] = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 500,
                "temperature": 0.2,
                "return_full_text": False,
            },
        }
        async with httpx.AsyncClient(timeout=self.settings.llm_timeout_sec) as client:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()

        if isinstance(data, list) and data and "generated_text" in data[0]:
            return data[0]["generated_text"]
        if isinstance(data, dict) and "generated_text" in data:
            return data["generated_text"]
        raise RuntimeError("Unexpected HuggingFace response format")

    def _extract_json(self, text: str) -> Dict[str, Any]:
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in model output")

        return json.loads(match.group(0))


def clamp_confidence(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, number))
