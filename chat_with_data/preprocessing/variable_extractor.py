"""
变量提取器
从研究主题中提取核心变量和控制变量
"""
from typing import List, Dict, Tuple
from common.llm_call import LLMAPIHandler
from common.config import OPENAI_API_KEY

# 初始化LLM处理器
llm_handler = LLMAPIHandler(openai_api_key=OPENAI_API_KEY)


def extract_core_variables(query: str) -> List[str]:
    """
    从用户查询中提取2-3个核心变量

    Args:
        query: 用户的研究查询

    Returns:
        核心变量列表（2-3个）

    Example:
        >>> query = "研究城市化对居民幸福感的影响"
        >>> variables = extract_core_variables(query)
        >>> print(variables)
        ['城市化水平', '居民幸福感']
    """
    prompt = f"""<TASK>
你是一个社会科学研究专家。请从以下研究问题中提取2-3个核心变量。
</TASK>

<RESEARCH_QUESTION>
{query}
</RESEARCH_QUESTION>

<EXTRACTION_GUIDELINES>
1. **核心变量定义**：
   - 自变量（Independent Variable）：研究者操纵或观察的变量
   - 因变量（Dependent Variable）：受自变量影响的结果变量
   - 中介/调节变量（如果明确提及）

2. **提取原则**：
   - 只提取最核心的2-3个变量
   - 使用简洁、专业的术语
   - 避免过于笼统的概念
   - 优先提取因果关系中的变量

3. **示例**：
   - 查询："教育水平对收入的影响"
   - 核心变量：["教育水平", "收入"]

   - 查询："社交媒体使用时长、心理健康与睡眠质量的关系"
   - 核心变量：["社交媒体使用时长", "心理健康", "睡眠质量"]
</EXTRACTION_GUIDELINES>

<OUTPUT_FORMAT>
请直接输出变量列表，每行一个，格式如下：
变量1
变量2
变量3
</OUTPUT_FORMAT>

核心变量："""

    try:
        response = llm_handler.call_llm(
            provider="openai",
            prompt=prompt,
            model="gpt-4o-mini",
            temperature=0.3,
            max_tokens=200
        )

        # 解析响应
        variables = []
        for line in response.strip().split('\n'):
            line = line.strip()
            # 移除可能的序号、破折号等
            line = line.lstrip('0123456789.-、 ')
            if line:
                variables.append(line)

        # 限制为2-3个变量
        variables = variables[:3]

        if len(variables) < 2:
            print(f"警告：只提取到{len(variables)}个核心变量，建议至少2个")

        return variables

    except Exception as e:
        print(f"提取核心变量时出错: {e}")
        return []


def identify_control_variables(
        query: str,
        core_variables: List[str]
) -> List[str]:
    """
    基于研究问题和核心变量，识别需要控制的变量

    Args:
        query: 用户的研究查询
        core_variables: 已提取的核心变量列表

    Returns:
        控制变量列表（5-8个）

    Example:
        >>> query = "研究教育对收入的影响"
        >>> core_vars = ["教育水平", "收入"]
        >>> control_vars = identify_control_variables(query, core_vars)
        >>> print(control_vars)
        ['年龄', '性别', '工作经验', '行业', '地区']
    """
    core_vars_text = "、".join(core_variables)

    prompt = f"""<TASK>
你是一个社会科学研究方法专家。请为以下研究问题识别需要控制的变量。
</TASK>

<RESEARCH_QUESTION>
{query}
</RESEARCH_QUESTION>

<CORE_VARIABLES>
{core_vars_text}
</CORE_VARIABLES>

<CONTROL_VARIABLE_GUIDELINES>
1. **控制变量定义**：
   - 可能影响因变量的其他因素
   - 需要在分析中控制其影响，以准确估计核心变量的效应

2. **识别原则**：
   - 基于理论和常识，识别可能的混淆变量
   - 考虑人口统计学变量（年龄、性别、教育等）
   - 考虑社会经济变量（收入、职业、地区等）
   - 考虑特定领域的相关变量
   - 提供5-8个控制变量

3. **示例**：
   - 研究问题："教育对收入的影响"
   - 核心变量："教育水平", "收入"
   - 控制变量：["年龄", "性别", "工作经验", "行业", "地区", "婚姻状况", "健康状况"]
</CONTROL_VARIABLE_GUIDELINES>

<OUTPUT_FORMAT>
请直接输出控制变量列表，每行一个，格式如下：
变量1
变量2
变量3
...
</OUTPUT_FORMAT>

控制变量："""

    try:
        response = llm_handler.call_llm(
            provider="openai",
            prompt=prompt,
            model="gpt-4o-mini",
            temperature=0.3,
            max_tokens=300
        )

        # 解析响应
        variables = []
        for line in response.strip().split('\n'):
            line = line.strip()
            # 移除可能的序号、破折号等
            line = line.lstrip('0123456789.-、 ')
            if line and line not in core_variables:  # 避免与核心变量重复
                variables.append(line)

        # 限制为5-8个变量
        if len(variables) > 8:
            variables = variables[:8]
        elif len(variables) < 5:
            print(f"警告：只识别到{len(variables)}个控制变量，建议5-8个")

        return variables

    except Exception as e:
        print(f"识别控制变量时出错: {e}")
        return []


def extract_variables(query: str) -> Tuple[List[str], List[str]]:
    """
    一次性提取核心变量和控制变量

    Args:
        query: 用户的研究查询

    Returns:
        (核心变量列表, 控制变量列表)
    """
    # 提取核心变量
    core_variables = extract_core_variables(query)

    if not core_variables:
        print("未能提取到核心变量")
        return [], []

    # 识别控制变量
    control_variables = identify_control_variables(query, core_variables)

    print(f"提取到核心变量: {core_variables}")
    print(f"识别到控制变量: {control_variables}")

    return core_variables, control_variables