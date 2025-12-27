"""
数据总结器
对检索到的数据和指标进行总结
"""
from typing import List, Dict, Any
from common.llm_call import LLMAPIHandler
from common.config import OPENAI_API_KEY

# 初始化LLM处理器
llm_handler = LLMAPIHandler(openai_api_key=OPENAI_API_KEY)


def summarize_data(metadata: List[Dict[str, Any]]) -> str:
    """
    总结AI相关指标数据，生成关于各指标所代表变量的总结报告

    Args:
        metadata: 指标元数据列表，格式为：
            [
                {
                    "indicator_name": "指标名称",
                    "description": "指标描述",
                    "unit": "统计单位",
                    "source": "数据来源",
                    "variable_type": "core/control"  # 可选
                },
                ...
            ]

    Returns:
        总结文本，区分核心指标和辅助指标

    Example:
        >>> metadata = [
        ...     {
        ...         "indicator_name": "人均GDP",
        ...         "description": "地区人均国内生产总值",
        ...         "unit": "元",
        ...         "source": "统计年鉴",
        ...         "variable_type": "core"
        ...     },
        ...     {
        ...         "indicator_name": "人口密度",
        ...         "description": "每平方公里人口数",
        ...         "unit": "人/平方公里",
        ...         "source": "人口普查",
        ...         "variable_type": "control"
        ...     }
        ... ]
        >>> summary = summarize_data(metadata)
    """
    if not metadata or not isinstance(metadata, list):
        return "暂无数据指标信息"

    # 分类指标
    core_indicators = []
    control_indicators = []

    for item in metadata:
        var_type = item.get("variable_type", "")
        if var_type == "core":
            core_indicators.append(item)
        else:
            control_indicators.append(item)

    # 如果没有分类，则全部视为核心指标
    if not core_indicators and not control_indicators:
        core_indicators = metadata

    # 构建指标文本
    def format_indicators(indicators: List[Dict]) -> str:
        text = ""
        for idx, item in enumerate(indicators, 1):
            name = item.get("indicator_name", "未知指标")
            desc = item.get("description", "")
            unit = item.get("unit", "")
            source = item.get("source", "")

            text += f"\n{idx}. {name}\n"
            if desc:
                text += f"   描述: {desc}\n"
            if unit:
                text += f"   单位: {unit}\n"
            if source:
                text += f"   来源: {source}\n"
        return text

    core_text = format_indicators(core_indicators) if core_indicators else "无"
    control_text = format_indicators(control_indicators) if control_indicators else "无"

    prompt = f"""<TASK>
你是一个数据分析专家。请基于以下数据指标信息，生成一个简洁的总结报告。
</TASK>

<CORE_INDICATORS>
核心研究指标:
{core_text}
</CORE_INDICATORS>

<CONTROL_INDICATORS>
控制变量指标:
{control_text}
</CONTROL_INDICATORS>

<SUMMARY_GUIDELINES>
1. **内容要求**:
   - 说明数据集包含的核心变量和控制变量
   - 概述各指标的测量维度
   - 说明数据来源和统计单位
   - 指出数据的潜在研究价值

2. **结构要求**:
   - 第一段：整体概述（数据集包含哪些方面的指标）
   - 第二段：核心指标说明
   - 第三段：控制变量说明（如果有）
   - 第四段：数据特征和研究价值

3. **语言要求**:
   - 使用专业、学术的语言
   - 保持客观中立的语气
   - 长度控制在150-250字
   - 避免列举式表达，使用流畅的段落

4. **示例**:
   本数据集整合了城市经济发展与居民生活质量相关的多维度指标。核心研究指标包括人均GDP、居民可支配收入等经济发展指标，以及幸福指数、生活满意度等主观福祉指标，能够全面反映经济发展与居民福祉的关系。控制变量涵盖人口密度、教育水平、医疗资源等社会经济特征，有助于排除混淆因素的影响。数据主要来源于官方统计年鉴，具有较高的权威性和可靠性，适合开展城市发展质量的实证研究。
</SUMMARY_GUIDELINES>

<OUTPUT_FORMAT>
直接输出总结内容，不要标题或其他说明文字。
</OUTPUT_FORMAT>

总结："""

    try:
        summary = llm_handler.call_llm(
            provider="openai",
            prompt=prompt,
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=400
        )

        return summary.strip()

    except Exception as e:
        print(f"生成数据总结时出错: {e}")
        return "数据总结生成失败"