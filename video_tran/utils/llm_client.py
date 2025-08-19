# -*- coding: utf-8 -*-
"""
@author: Gemini
@software: PyCharm
@file: llm_client.py
@time: 2025/8/15 17:15
"""
import os
import requests
import asyncio
import aiohttp
from typing import Dict, Any


class DeepSeekClient:
    """
    用于与 DeepSeek API 交互的客户端。
    支持同步和异步请求。
    """

    def __init__(self, api_key: str = None, max_retries: int = 3, timeout: int = 20):
        """
        初始化 DeepSeekClient。

        Args:
            api_key (str, optional): DeepSeek API 密钥。如果未提供，
                                     将尝试从环境变量 DEEPSEEK_TOKEN 读取。
            max_retries (int): API 请求失败时的最大重试次数。
            timeout (int): 请求超时时间（秒）。
        """
        self.api_key = api_key or os.getenv("DEEPSEEK_TOKEN")
        if not self.api_key:
            raise ValueError("未提供 DeepSeek API 密钥，也未在环境变量 DEEPSEEK_TOKEN 中找到。")
        self.api_url = "https://api.deepseek.com/v1/chat/completions"
        self.max_retries = max_retries
        self.timeout = timeout
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def _make_request(self, payload: dict) -> dict:
        """
        (同步) 向 DeepSeek API 发送请求。
        """
        for attempt in range(self.max_retries):
            try:
                response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=self.timeout)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                print(f"请求 DeepSeek API 时出错 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    return None
        return None

    async def _make_async_request(self, session: aiohttp.ClientSession, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        (异步) 向 DeepSeek API 发送请求。
        """
        for attempt in range(self.max_retries):
            try:
                async with session.post(self.api_url, headers=self.headers, json=payload, timeout=self.timeout) as response:
                    response.raise_for_status()
                    return await response.json()
            except aiohttp.ClientError as e:
                print(f"异步请求 DeepSeek API 时出错 (尝试 {attempt + 1}/{self.max_retries}): {e}")
                if attempt == self.max_retries - 1:
                    return None
                await asyncio.sleep(1)  # 在重试前稍作等待
        return None

    def correct_text(self, text: str) -> str:
        """
        (同步) 使用 LLM 校正 ASR 文本。
        """
        prompt = self._get_correction_prompt(text)
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "你是一个专业的速记员和文本编辑。"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
        }
        
        response_data = self._make_request(payload)
        if response_data and response_data.get("choices"):
            return response_data["choices"][0]["message"]["content"].strip()
        
        return text

    def translate_text(self, text: str, target_lang: str, duration: float) -> str:
        """
        (同步) 使用 LLM 翻译文本，并考虑原始时长。
        """
        prompt = self._get_translation_prompt(text, target_lang, duration)
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": f"You are an expert translator specializing in video dubbing. You will translate text into {target_lang}."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
        }
        
        response_data = self._make_request(payload)
        if response_data and response_data.get("choices"):
            return response_data["choices"][0]["message"]["content"].strip()
        
        return ""

    async def correct_text_async(self, session: aiohttp.ClientSession, text: str) -> str:
        """
        (异步) 使用 LLM 校正 ASR 文本。
        """
        prompt = self._get_correction_prompt(text)
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": "你是一个专业的速记员和文本编辑。"},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
        }
        response_data = await self._make_async_request(session, payload)
        if response_data and response_data.get("choices"):
            return response_data["choices"][0]["message"]["content"].strip()
        return text

    async def translate_text_async(self, session: aiohttp.ClientSession, text: str, target_lang: str, duration: float) -> str:
        """
        (异步) 使用 LLM 翻译文本，并考虑原始时长。
        """
        prompt = self._get_translation_prompt(text, target_lang, duration)
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": f"You are an expert translator specializing in video dubbing. You will translate text into {target_lang}."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
        }
        response_data = await self._make_async_request(session, payload)
        if response_data and response_data.get("choices"):
            return response_data["choices"][0]["message"]["content"].strip()
        return ""

    @staticmethod
    def _get_correction_prompt(text: str) -> str:
        return f"""
你是一个专业的视频字幕编辑。请对以下由ASR(自动语音识别)生成的、用于视频字幕的文本进行校正和优化。
你的任务是严格遵守视频字幕的标点规范：
1.  **修正识别错误**：修正文本中明显的语音识别错误
2.  **优化标点符号，遵守以下规则**:
    *   **标点符号格式**：所有标点符号都必须使用半角字符
    *   **句末**：句末绝对不能使用句号 `.` 或逗号 `,`
    *   **必须保留**：必须保留问号 `?` 和感叹号 `!`
    *   **停顿**：长句内部的停顿可以用逗号 `,`，但更推荐使用空格来替代，以保持字幕简洁
    *   **省略号**：表示对话中断或延续时，使用标准的省略号 `…` (一个字符)
    *   **引号**：仅在绝对必要时（如直接引语）才使用引号 `""`
3.  **保持口语化**：保留原始的口语风格，不要进行书面化改写
4.  **纯文本输出**：不要添加任何与原始文本无关的内容、评论或解释。直接返回优化后的文本

原始文本：
"{text}"

优化后的文本：
"""

    @staticmethod
    def _get_translation_prompt(text: str, target_lang: str, duration: float) -> str:
        return f"""
You are an expert translator specializing in video dubbing. Your task is to translate the following text into {target_lang}.

Please adhere to the following rules:
1.  **Naturalness**: The translation must be fluent and sound natural in {target_lang}, as if it were originally spoken.
2.  **Lip-Sync Friendly**: The original audio for this text was {duration:.2f} seconds long. Your translation should be of a comparable length when spoken naturally. Avoid overly long or very short translations.
3.  **Contextual Accuracy**: Maintain the original meaning, tone, and nuance.
4.  **Output**: Return only the translated text, without any extra explanations, comments, or quotation marks.

Source Text:
"{text}"

Translated Text:
"""
