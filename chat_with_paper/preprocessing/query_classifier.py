"""
查询分类器
对用户查询进行分类
"""
from common.llm_call import handler


def classify_query_type(query: str) -> int:
    """
    判断用户查询属于哪种类型

    Args:
        query: 用户查询内容

    Returns:
        0表示QA型，1表示数据型，2表示混合型
    """
    try:
        prompt = f"""
<TASK>
你是一个精确的查询分类器。请根据<CLASSIFICATION_FRAMEWORK>中定义的标准，将用户查询分类为以下三种类型之一：QA型(0)、数据型(1)或混合型(2)。
</TASK>

<QUERY>
{query}
</QUERY>

<CLASSIFICATION_FRAMEWORK>
## 类型定义
0. QA型（纯问答）：用户请求解释、定义、方法指导或简单事实确认，答案通过文本直接完整表达，无需结构化数据支持。
1. 数据型（需Excel表格）：用户明确要求结构化、多维度、机器可处理的指标、指数、数据集合，需通过表格呈现原始数据或统计分析结果。
2. 混合型（需文本+表格）：用户在同一问题中同时请求解释性答案和结构化数据，需文本回答与数据表格配合使用才能完整满足需求。

## 判断标准
### QA型(0)核心特征：
- 问题聚焦于"是什么/为什么/如何做"
- 答案不依赖数值或多维度数据组合
- 无需机器可解析的数据组织形式

### 数据型(1)核心特征：
- 包含具体指标（销售额、增长率等）
- 指定多维度组合（时间+地区+品类等）
- 要求完整数据集（而非单一数值或总结性描述）
- 使用导出/生成/提供/统计等动词

### 混合型(2)核心特征：
- 问题包含双层意图（先解释概念，再要求数据验证）
- 答案需要逻辑关联（文字结论基于数据支撑）
- 数据表格是文本分析的必要补充
- 通常含逻辑连接词（并/同时/以及）
</CLASSIFICATION_FRAMEWORK>

<OUTPUT_INSTRUCTION>
只返回分类结果数字：0、1或2，不要返回任何其他内容，也无需解释理由。
</OUTPUT_INSTRUCTION>

分类结果："""

        try:
            result = handler.call_llm(
                provider="zhipuai",
                prompt=prompt,
                model="glm-4-airx",
                max_tokens=20,
                temperature=0.7
            )
            if "API call failed" in str(result):
                raise Exception("API call failed detected in response")
        except:
            result = handler.call_llm(
                provider="openai",
                prompt=prompt,
                model="gpt-4o-mini",
                max_tokens=20,
                temperature=0.7
            )

        if result == "0":
            return 0
        elif result == "1":
            return 1
        elif result == "2":
            return 2
        else:
            return 0

    except Exception as e:
        print(f"调用大模型分类查询失败: {e}")
        return 0


def classify_academic_query_type(query: str) -> int:
    """
    判断学术查询属于哪种类型

    Args:
        query: 用户查询内容

    Returns:
        0表示行业现状相关，1表示理论相关，2表示方法相关，
        3表示科学事实相关，4表示研究进展相关
    """
    try:
        prompt = f"""
<TASK>
你是一个专业的学术查询分类器。请根据<CLASSIFICATION_FRAMEWORK>中定义的标准，将用户的学术查询准确分类为以下五种类型之一：
行业现状相关(0)、理论相关(1)、方法相关(2)、科学事实相关(3)或研究进展相关(4)。
</TASK>

<QUERY>{query}</QUERY>

<CLASSIFICATION_FRAMEWORK>
## 类型定义
0. 行业现状相关：探讨现实背景、政策环境、产业动态等应用层面的情境信息。
1. 理论相关：涉及理论基础、分析框架、研究假设等抽象知识体系。
2. 方法相关：询问研究方法、数据采集、测量工具、分析技术等实证操作细节。
3. 科学事实相关：求证研究结论、已验证的因果关系、可复现的实验结果等确定性知识。
4. 研究进展相关：评估学术创新性、研究空白、学术争议、未来方向等知识发展状态。

## 判断标准
### 行业现状相关(0)核心特征：
- 聚焦实践层面的现象描述
- 时间指向为当前或近期
- 涉及动态变化的信息，需更新数据
- 常见关键词：现状、趋势、政策、挑战、机遇、竞争格局、市场规模

### 理论相关(1)核心特征：
- 聚焦抽象理论层面
- 无明确时间限定性
- 可能存在学派分歧
- 常见关键词：理论模型、框架、假设、机制、关系、原理、学派

### 方法相关(2)核心特征：
- 聚焦操作层面的研究流程
- 具有方法论通用性
- 涉及有条件的最佳实践
- 常见关键词：方法、样本、数据源、量表、信效度、实验设计、回归模型

### 科学事实相关(3)核心特征：
- 聚焦已完成研究的实证结果
- 高确定性内容，需引证来源
- 关注可验证的因果关系或影响机制
- 常见关键词：结论、证明、证实、相关、影响、效应、作用

### 研究进展相关(4)核心特征：
- 聚焦近期或未来的知识发展
- 时间指向为近期或未来趋势
- 涉及开放性学术讨论
- 常见关键词：创新点、局限性、学术争议、研究前沿、未来方向、知识缺口
</CLASSIFICATION_FRAMEWORK>

<OUTPUT_INSTRUCTION>
只返回分类结果数字：0、1、2、3或4，不要返回任何其他内容，也无需解释理由。
</OUTPUT_INSTRUCTION>

分类结果："""

        try:
            result = handler.call_llm(
                provider="zhipuai",
                prompt=prompt,
                model="glm-4-airx",
                max_tokens=5,
                temperature=0.7
            )
            if "API call failed" in str(result):
                raise Exception("API call failed detected in response")
        except:
            result = handler.call_llm(
                provider="openai",
                prompt=prompt,
                model="gpt-4o-mini",
                max_tokens=5,
                temperature=0.7
            )

        if result in ["0", "1", "2", "3", "4"]:
            return int(result)
        else:
            return 3

    except Exception as e:
        print(f"调用大模型分类学术查询失败: {e}")
        return 3