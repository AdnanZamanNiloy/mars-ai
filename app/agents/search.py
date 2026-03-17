import asyncio
from typing import Any, Dict, List

import httpx
from duckduckgo_search import DDGS

from app.core.config import Settings


class SearchClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.semaphore = asyncio.Semaphore(max(1, settings.max_parallel_search))

    async def run_search(self, sub_questions: List[str]) -> List[Dict[str, str]]:
        tasks = [self._bounded_search(question) for question in sub_questions]
        batches = await asyncio.gather(*tasks)

        results: List[Dict[str, str]] = []
        for question, items in zip(sub_questions, batches):
            for item in items:
                item["sub_question"] = question
                results.append(item)
        return results

    async def _bounded_search(self, question: str) -> List[Dict[str, str]]:
        async with self.semaphore:
            return await self._retry_search(question)

    async def _retry_search(self, question: str, retries: int = 3) -> List[Dict[str, str]]:
        for attempt in range(retries):
            try:
                if self.settings.tavily_api_key:
                    tavily_results = await self._tavily_search(question)
                    if tavily_results:
                        return tavily_results
                ddg_results = await self._duckduckgo_search(question)
                return ddg_results
            except Exception:
                if attempt == retries - 1:
                    return []
                await asyncio.sleep(0.7 * (attempt + 1))
        return []

    async def _tavily_search(self, question: str) -> List[Dict[str, str]]:
        payload: Dict[str, Any] = {
            "api_key": self.settings.tavily_api_key,
            "query": question,
            "search_depth": "basic",
            "max_results": 4,
        }
        async with httpx.AsyncClient(timeout=self.settings.search_timeout_sec) as client:
            response = await client.post("https://api.tavily.com/search", json=payload)
            response.raise_for_status()
            data = response.json()

        items = data.get("results", []) if isinstance(data, dict) else []
        return [
            {
                "title": str(item.get("title", "")).strip(),
                "url": str(item.get("url", "")).strip(),
                "snippet": str(item.get("content", "")).strip(),
            }
            for item in items
            if item.get("url")
        ]

    async def _duckduckgo_search(self, question: str) -> List[Dict[str, str]]:
        def _search() -> List[Dict[str, str]]:
            with DDGS() as ddgs:
                rows = list(ddgs.text(question, max_results=4))
            return [
                {
                    "title": str(row.get("title", "")).strip(),
                    "url": str(row.get("href", "")).strip(),
                    "snippet": str(row.get("body", "")).strip(),
                }
                for row in rows
                if row.get("href")
            ]

        return await asyncio.to_thread(_search)
