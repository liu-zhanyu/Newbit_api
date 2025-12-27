"""
查询优化器
生成优化的检索词
"""
import re
from common.llm_call import handler


def is_english_text(text: str) -> bool:
    """判断文本是否为英文"""
    cleaned_text = ''.join(c for c in text if c.isalnum() or c.isspace())
    if not cleaned_text:
        return False

    english_count = 0
    chinese_count = 0

    for char in cleaned_text:
        if 'a' <= char.lower() <= 'z':
            english_count += 1
        elif '\u4e00' <= char <= '\u9fff':
            chinese_count += 1

    total_relevant_chars = english_count + chinese_count
    if total_relevant_chars == 0:
        return False

    english_ratio = english_count / total_relevant_chars
    return english_ratio > 0.7


def get_optimal_query(query: str) -> str:
    """
    生成优化的检索词（与原query同语言）

    Args:
        query: 原始查询文本

    Returns:
        优化后的检索词
    """
    is_english = is_english_text(query)
    source_language = "英文" if is_english else "中文"

    prompt = f"""
你是一名专业的社会科学研究人员，请为<用户查询文本>生成二到四个语言为{source_language}的文献检索词，在生成过程中，严格遵守<检索词要求>，并且按照<输出结构>输出翻译后的查询文本。

<检索词要求>
1. 提取核心学术概念并转化为专业检索词
2. 保持专业术语准确
3. 检索词之间采用空格连接，不要包含逗号等标点符号，单词（token）数在四个以下
4. 检索词中仅纳入专业词汇，请勿加入作者、年份、文献语言、期刊名等筛选信息
</检索词要求>

<用户查询文本>
{query}
</用户查询文本>

<输出结构>
输出时用[TRANS_START]和[TRANS_END]包裹结果，仅返回由[TRANS_START]和[TRANS_END]包裹的翻译结果，不要输出其它任何内容
</输出结构>

你生成的检索词({source_language})：
"""

    try:
        response_content = handler.call_llm(
            provider="zhipuai",
            prompt=prompt,
            model="glm-4-airx",
            max_tokens=100,
            temperature=0.7
        )
        if "API call failed" in str(response_content):
            raise Exception("API call failed detected in response")
    except:
        response_content = handler.call_llm(
            provider="openai",
            prompt=prompt,
            model="gpt-4o-mini",
            max_tokens=100,
            temperature=0.7
        )

    translation_pattern = r'\[TRANS_START\]([\s\S]*)\[TRANS_END\]'
    match = re.search(translation_pattern, response_content)

    if match:
        translated_query = match.group(1)
    else:
        translated_query = response_content.replace('[TRANS_START]', '').replace('[TRANS_END]', '')

    return translated_query