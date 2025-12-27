"""
LLM API调用处理器
支持多个LLM提供商：OpenAI、Claude、ARK、SiliconFlow、ZhipuAI、Qwen、DeepSeek
"""
from typing import Optional, Union, Generator, Dict, List
import anthropic
import requests
from openai import OpenAI
from zhipuai import ZhipuAI
import json


class LLMAPIHandler:
    def __init__(self,
                 openai_api_key: Optional[str] = None,
                 anthropic_api_key: Optional[str] = None,
                 ark_api_key: Optional[str] = None,
                 siliconflow_api_key: Optional[str] = None,
                 zhipuai_api_key: Optional[str] = None,
                 qwen_api_key: Optional[str] = None,
                 deepseek_api_key: Optional[str] = None):
        """
        初始化各个LLM提供商的API客户端

        Args:
            openai_api_key: OpenAI API密钥
            anthropic_api_key: Claude API密钥
            ark_api_key: 字节ARK API密钥
            siliconflow_api_key: SiliconFlow API密钥
            zhipuai_api_key: 智谱AI API密钥
            qwen_api_key: 通义千问API密钥
            deepseek_api_key: DeepSeek API密钥
        """
        # 初始化各个客户端
        if openai_api_key:
            self.openai_client = OpenAI(api_key=openai_api_key)

        if anthropic_api_key:
            self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)

        if ark_api_key:
            self.ark_client = OpenAI(
                base_url="https://ark.cn-beijing.volces.com/api/v3",
                api_key=ark_api_key,
            )

        if siliconflow_api_key:
            self.siliconflow_api_key = siliconflow_api_key
            self.siliconflow_base_url = "https://api.siliconflow.cn/v1"

        if zhipuai_api_key:
            self.zhipuai_client = ZhipuAI(api_key=zhipuai_api_key)

        if qwen_api_key:
            self.qwen_client = OpenAI(
                api_key=qwen_api_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )

        if deepseek_api_key:
            self.deepseek_client = OpenAI(
                api_key=deepseek_api_key,
                base_url="https://api.deepseek.com"
            )

        # 默认模型配置
        self.default_models = {
            'openai': 'gpt-4o-mini',
            'claude': 'claude-sonnet-4-20250514',
            'ark': 'doubao-1-5-lite-32k-250115',
            'siliconflow': 'deepseek-ai/DeepSeek-V3',
            'zhipuai': 'glm-4-air',
            'qwen': 'qwen-turbo-2025-04-28',
            'deepseek': 'deepseek-chat'
        }

    # ... 其他方法保持不变 ...

    def call_llm(self,
                 provider: str,
                 prompt: str,
                 model: Optional[str] = None,
                 max_tokens: Optional[int] = None,
                 temperature: Optional[float] = None,
                 stream: bool = False,
                 system_prompt: Optional[str] = None,
                 history: Optional[List[Dict[str, str]]] = None) -> Union[str, Generator[str, None, None]]:
        """
        统一的LLM调用方法，支持多个提供商和流式输出
        """
        provider = provider.lower()

        # 设置默认参数
        params = {
            'prompt': prompt,
            'model': model or self.default_models.get(provider),
            'max_tokens': max_tokens or 1000,
            'stream': stream,
            'system_prompt': system_prompt,
            'history': history
        }

        # 添加temperature参数（Claude除外）
        if provider != 'claude':
            params['temperature'] = temperature or 0.7

        # 调用对应提供商的方法
        if provider == 'openai':
            return self.call_openai(**params)
        elif provider == 'claude':
            return self.call_claude(**params)
        elif provider == 'ark':
            return self.call_ark(**params)
        elif provider == 'siliconflow':
            return self.call_siliconflow(**params)
        elif provider == 'zhipuai':
            return self.call_zhipuai(**params)
        elif provider == 'qwen':
            return self.call_qwen(**params)
        elif provider == 'deepseek':
            return self.call_deepseek(**params)
        else:
            return f"Unsupported provider: {provider}"


# 从配置文件导入API Keys并初始化全局handler
from common.config import (
    OPENAI_API_KEY,
    CLAUDE_API_KEY,
    ARK_API_KEY,
    SILICONFLOW_API_KEY,
    ZHIPU_API_KEY,
    QWEN_API_KEY,
    DEEPSEEK_API_KEY
)

# 初始化全局handler实例
handler = LLMAPIHandler(
    openai_api_key=OPENAI_API_KEY,
    anthropic_api_key=CLAUDE_API_KEY,
    ark_api_key=ARK_API_KEY,
    siliconflow_api_key=SILICONFLOW_API_KEY,
    zhipuai_api_key=ZHIPU_API_KEY,
    qwen_api_key=QWEN_API_KEY,
    deepseek_api_key=DEEPSEEK_API_KEY
)