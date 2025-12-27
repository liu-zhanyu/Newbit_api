"""
指标推荐器
为每个变量推荐具体的测量指标
"""
from typing import List, Dict
from common.llm_call import LLMAPIHandler
from common.config import OPENAI_API_KEY

# 初始化LLM处理器
llm_handler = LLMAPIHandler(openai_api_key=OPENAI_API_KEY)


def suggest_measurement_indicators(
        variables: List[str],
        context: str = ""
) -> Dict[str, List[str]]:
    """
    为每个变量推荐3个具体的测量指标

    Args:
        variables: 变量名称列表
        context: 研究背景或上下文（可选）

    Returns:
        变量到指标列表的映射字典，每个变量对应3个指标

    Example:
        >>> variables = ["教育水平", "收入", "健康状况"]
        >>> indicators = suggest_measurement_indicators(variables)
        >>> print(indicators)
        {
            "教育水平": ["受教育年限", "最高学历", "职业培训时长"],
            "收入": ["年收入", "月收入", "家庭人均收入"],
            "健康状况": ["自评健康", "慢性病数量", "BMI指数"]
        }
    """
    if not variables:
        return {}

    variables_text = "\n".join([f"{i + 1}. {var}" for i, var in enumerate(variables)])
    context_section = f"\n<RESEARCH_CONTEXT>\n{context}\n</RESEARCH_CONTEXT>\n" if context else ""

    prompt = f"""<TASK>
你是一个社会科学研究方法专家。请为以下变量推荐具体的测量指标。
</TASK>

<VARIABLES>
{variables_text}
</VARIABLES>
{context_section}
<INDICATOR_GUIDELINES>
1. **测量指标要求**：
   - 为每个变量推荐3个具体的测量指标
   - 指标应该是可观察、可测量的
   - 指标名称应该简洁、专业
   - 优先推荐常用的、标准化的指标

2. **指标类型**：
   - 客观指标：可以直接测量的数值（如年龄、收入、教育年限）
   - 主观指标：通过问卷或量表测量的（如满意度、自评健康）
   - 复合指标：由多个维度组成的（如BMI、幸福指数）

3. **示例**：
   - 变量："教育水平"
   - 指标：
     1. 受教育年限（年）
     2. 最高学历（小学/初中/高中/本科/研究生）
     3. 是否有职业资格证书（是/否）

   - 变量："收入"
   - 指标：
     1. 个人年收入（元）
     2. 家庭年收入（元）
     3. 相对收入水平（低/中/高）

4. **注意事项**：
   - 指标应与变量概念紧密相关
   - 避免过于复杂或难以获取的指标
   - 考虑指标的实际可行性
</INDICATOR_GUIDELINES>

<OUTPUT_FORMAT>
请以JSON格式输出，格式如下：
{{
    "变量1": [
        "指标1",
        "指标2",
        "指标3"
    ],
    "变量2": [
        "指标1",
        "指标2",
        "指标3"
    ]
}}
</OUTPUT_FORMAT>

测量指标推荐："""

    try:
        response = llm_handler.call_llm(
            provider="openai",
            prompt=prompt,
            model="gpt-4o-mini",
            temperature=0.3,
            max_tokens=1000
        )

        # 解析JSON
        import json
        response = response.replace("```json", "").replace("```", "").strip()
        indicators_dict = json.loads(response)

        # 验证每个变量是否有3个指标
        result = {}
        for var in variables:
            if var in indicators_dict:
                indicators = indicators_dict[var]
                if isinstance(indicators, list):
                    # 确保有3个指标
                    if len(indicators) < 3:
                        print(f"警告：变量 '{var}' 只有 {len(indicators)} 个指标")
                    result[var] = indicators[:3]  # 只取前3个
                else:
                    print(f"警告：变量 '{var}' 的指标格式不正确")
                    result[var] = []
            else:
                print(f"警告：未找到变量 '{var}' 的指标")
                result[var] = []

        return result

    except json.JSONDecodeError as e:
        print(f"解析指标JSON时出错: {e}")
        return {var: [] for var in variables}
    except Exception as e:
        print(f"推荐测量指标时出错: {e}")
        return {var: [] for var in variables}


def suggest_indicators_for_variable(
        variable: str,
        num_indicators: int = 3,
        context: str = ""
) -> List[str]:
    """
    为单个变量推荐指标

    Args:
        variable: 变量名称
        num_indicators: 需要的指标数量（默认3个）
        context: 研究背景或上下文（可选）

    Returns:
        指标列表
    """
    result = suggest_measurement_indicators([variable], context)
    return result.get(variable, [])