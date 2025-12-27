"""
查询改写器
对查询进行改写和消歧
"""
import re
from typing import List, Dict
from common.llm_call import handler


def rewrite_and_disambiguate_query(history: List[Dict], query: str) -> str:
    """
    对用户查询进行改写和消除歧义

    Args:
        history: 历史对话记录，格式为[{"content": "文本内容", "role": "user/assistant"}]
        query: 当前用户查询内容

    Returns:
        改写并消除歧义后的查询
    """
    try:
        # 格式化历史记录
        history_text = ""
        for message in history:
            role = message.get("role", "")
            content = message.get("content", "")
            if role == "user":
                history_text += f"用户: {content}\n"
            elif role == "assistant":
                history_text += f"助手: {content}\n"

        prompt = f"""<TASK>
你是一个专业的检索查询重写专家。请根据用户的历史对话和当前查询，对查询进行改写和消除歧义，使其更加明确、具体、完整，并更适合用于检索系统。
</TASK>

<HISTORY>
{history_text}
</HISTORY>

<CURRENT_QUERY>
{query}
</CURRENT_QUERY>

<INSTRUCTIONS>
1. 改写后的查询的语言应当与<CURRENT_QUERY>保持一致
2. 分析用户的历史对话和当前查询，识别上下文信息和关键实体
3. 对当前查询进行改写，使其更加完整、明确、具体，添加有助于检索的关键信息
4. 解决查询中的指代不明或歧义问题，确保每个实体和概念都明确具体
5. 使用精确的术语和描述，增强查询的可检索性

注意：
- 确保改写后的查询包含足够的具体细节和关键词，以便优化检索效果
- 不要添加用户未提及的新需求或假设
- 不要改变用户查询的基本意图
- 保持改写后查询的具体性和明确性
</INSTRUCTIONS>

<OUTPUT_FORMAT>
只返回改写后的查询，不要包含任何解释或其他内容。
使用[QUERY_START]和[QUERY_END]标记改写后的查询。
</OUTPUT_FORMAT>

改写结果："""

        try:
            result = handler.call_llm(
                provider="zhipuai",
                prompt=prompt,
                model="glm-4-airx",
                max_tokens=100,
                temperature=0.7
            )
            if "API call failed" in str(result):
                raise Exception("API call failed detected in response")
        except:
            result = handler.call_llm(
                provider="openai",
                prompt=prompt,
                model="gpt-4o-mini",
                max_tokens=100,
                temperature=0.7
            )

        match = re.search(r'\[QUERY_START\](.*?)\[QUERY_END\]', result, re.DOTALL)

        if match:
            return match.group(1).strip()
        else:
            return query

    except Exception as e:
        print(f"调用大模型改写查询失败: {e}")
        return query