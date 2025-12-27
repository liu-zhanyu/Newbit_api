"""
主题分析器
分析研究主题，提取核心变量和控制变量
"""
from typing import Dict, Any, Optional, List
from common.llm_call import LLMAPIHandler
from common.config import OPENAI_API_KEY

# 初始化LLM处理器
llm_handler = LLMAPIHandler(openai_api_key=OPENAI_API_KEY)


class VariableExtractionAgent:
    """变量提取代理类"""

    def __init__(self, api_key: Optional[str] = None):
        """
        初始化变量提取代理

        Args:
            api_key: OpenAI API密钥，如果不提供则使用配置中的密钥
        """
        self.api_key = api_key or OPENAI_API_KEY
        self.llm_handler = LLMAPIHandler(openai_api_key=self.api_key)

    def ask_gpt(
            self,
            prompt: str,
            model: str = "gpt-4o-mini",
            temperature: float = 0.7,
            max_tokens: int = 1000
    ) -> str:
        """
        通用LLM调用方法

        Args:
            prompt: 提示词
            model: 模型名称
            temperature: 温度参数
            max_tokens: 最大token数

        Returns:
            LLM响应文本
        """
        try:
            response = self.llm_handler.call_llm(
                provider="openai",
                prompt=prompt,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.strip()
        except Exception as e:
            print(f"调用LLM时出错: {e}")
            return ""

    def analyze_research_topic(self, topic: str) -> Dict[str, Any]:
        """
        分析研究主题，提取研究领域、主要变量等信息

        Args:
            topic: 研究主题描述

        Returns:
            包含研究领域、核心问题、可能变量等信息的字典
        """
        prompt = f"""<TASK>
你是一个社会科学研究方法专家。请分析以下研究主题，识别其研究领域、核心问题和可能涉及的变量。
</TASK>

<RESEARCH_TOPIC>
{topic}
</RESEARCH_TOPIC>

<ANALYSIS_GUIDELINES>
1. 识别研究领域（如：经济学、社会学、心理学、管理学等）
2. 提炼核心研究问题（研究想要探讨什么）
3. 列出可能的自变量（independent variables）
4. 列出可能的因变量（dependent variables）
5. 识别潜在的中介变量或调节变量（如果适用）
6. 列出需要控制的变量
</ANALYSIS_GUIDELINES>

<OUTPUT_FORMAT>
请以JSON格式输出，包含以下字段：
{{
    "research_field": "研究领域",
    "core_question": "核心研究问题",
    "independent_variables": ["自变量1", "自变量2"],
    "dependent_variables": ["因变量1", "因变量2"],
    "mediator_moderator": ["中介/调节变量1"],
    "control_variables": ["控制变量1", "控制变量2"]
}}
</OUTPUT_FORMAT>

分析结果："""

        response = self.ask_gpt(prompt, temperature=0.3)

        # 尝试解析JSON
        import json
        try:
            # 移除可能的markdown代码块标记
            response = response.replace("```json", "").replace("```", "").strip()
            result = json.loads(response)
            return result
        except json.JSONDecodeError:
            print("无法解析LLM返回的JSON，返回空结果")
            return {
                "research_field": "",
                "core_question": "",
                "independent_variables": [],
                "dependent_variables": [],
                "mediator_moderator": [],
                "control_variables": []
            }

    def extract_variable_definitions(self, variables: List[str]) -> Dict[str, str]:
        """
        为变量生成定义和说明

        Args:
            variables: 变量名称列表

        Returns:
            变量名到定义的映射字典
        """
        if not variables:
            return {}

        variables_text = "\n".join([f"- {var}" for var in variables])

        prompt = f"""<TASK>
请为以下变量提供学术定义和测量说明。
</TASK>

<VARIABLES>
{variables_text}
</VARIABLES>

<GUIDELINES>
1. 每个变量提供简洁的学术定义（1-2句话）
2. 说明该变量通常如何测量
3. 使用客观、学术的语言
</GUIDELINES>

<OUTPUT_FORMAT>
请以JSON格式输出：
{{
    "变量名1": "定义和测量说明",
    "变量名2": "定义和测量说明"
}}
</OUTPUT_FORMAT>

定义："""

        response = self.ask_gpt(prompt, temperature=0.3)

        import json
        try:
            response = response.replace("```json", "").replace("```", "").strip()
            result = json.loads(response)
            return result
        except json.JSONDecodeError:
            print("无法解析变量定义JSON")
            return {}