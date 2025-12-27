"""
追问生成器
基于对话历史生成学术性追问
"""
import re
from typing import List, Dict
from common.llm_call import LLMAPIHandler
from common.config import OPENAI_API_KEY

# 初始化LLM处理器
llm_handler = LLMAPIHandler(openai_api_key=OPENAI_API_KEY)


def generate_followup_questions(dialogue_dict: Dict[str, str]) -> List[str]:
    """
    基于对话字典生成学术性追问

    Args:
        dialogue_dict: {
            "user": "合并后的所有用户发言内容",
            "assistant": "合并后的所有助手回复内容"
        }

    Returns:
        包含追问的字符串列表，如:
        ["追问内容1?", "追问内容2?"]

    Raises:
        ValueError: 当输入字典格式不正确时
    """
    # 输入验证
    if not isinstance(dialogue_dict, dict):
        raise ValueError("输入必须是字典类型")
    if "user" not in dialogue_dict or "assistant" not in dialogue_dict:
        raise ValueError("字典必须包含'user'和'assistant'键")
    if not isinstance(dialogue_dict["user"], str) or not isinstance(dialogue_dict["assistant"], str):
        raise ValueError("对话内容必须是字符串类型")

    # 默认追问（用于错误回退）
    DEFAULT_QUESTIONS = [
        "能否进一步阐述核心概念的操作化定义？",
        "该观点的实证支持来自哪些具体研究？"
    ]

    try:
        # 构建对话历史字符串
        history_dialogue = f"用户: {dialogue_dict['user']}\n助手: {dialogue_dict['assistant']}"

        prompt = f"""<TASK>
你是一个学术对话分析引擎，目标是通过精准追问推动讨论明晰化。请基于<HISTORY_DIALOGUE>中的对话内容，生成三个中性追问，严格遵循<QUESTION_GUIDELINES>中的规则。
</TASK>

<HISTORY_DIALOGUE>
{history_dialogue}
</HISTORY_DIALOGUE>

<QUESTION_GUIDELINES>
## 追问准则
1. 模糊点锚定策略：
   - 概念模糊：定位高频未定义术语（如"概念A的定义是什么？"）
   - 逻辑断层：识别未经验证的推论跳跃（如"从A变量影响B变量的内在机制是什么？"）
   - 数据缺口：发现未被量化的核心变量（如"概念A可以通过什么方法和数据来测量？"）

2. 追问生成原则：
   - 采用"概念界定→机制验证→实证路径"递进结构
   - 每个问题必须包含对话中的原文关键词（用单引号标注）
   - 使用客观限定表达式（"在X框架下"/"当控制Y变量时"/"哪些测量维度"）

3. 安全过滤机制：
   - 禁止输出以下类型的问题：
     * 包含政治敏感内容的问题
     * 涉及特定政策评价的问题
     * 针对特定群体的歧视性问题
     * 涉及现行制度批评的问题
     * 引导意识形态站队的问题
     * 涉及宗教信仰判断的问题
     * 暗示地缘政治立场的问题
     * 含有民族或区域偏见的问题
     * 讨论社会争议敏感事件的问题
     * 暗示经济体系优劣的问题
   - 发现敏感内容时，应生成完全中性的学术问题（如"不同约束条件下的表现差异"）

## 逻辑完整性检测
1. 构建论点树状图：
   - 主论点 → 子论点 → 支撑数据
2. 识别三类断裂点：
   - 红区：无数据支撑的终节点
   - 黄区：单一数据支撑多重推论
   - 蓝区：存在矛盾证据的节点
3. 生成对应追问：
   [FQ_START]支撑【子论点X】的数据是否排除竞争性解释？[FQ_END]

## 优质案例
历史对话中出现："领导风格影响员工创新行为"
[FQ_START]'领导风格'包含哪些具体维度？[FQ_END]
[FQ_START]领导风格影响创新行为的内在机理是什么？[FQ_END]
[FQ_START]领导风格可以通过什么方式来测量？[FQ_END]
</QUESTION_GUIDELINES>

<OUTPUT_INSTRUCTION>
1. 问题长度25-35汉字，以问号结尾
2. 每个追问用[FQ_START]和[FQ_END]严格包裹
3. 置于回答末尾，与正文间隔一个空行
4. 仅输出三个追问，不要添加任何解释或分析过程
</OUTPUT_INSTRUCTION>

追问结果："""

        # 调用LLM API
        response = llm_handler.call_llm(
            provider="openai",
            prompt=prompt,
            model="gpt-4o-mini",
            max_tokens=100,
            temperature=0.7
        )

        raw_output = response.strip()
        print(raw_output)

        # 使用正则表达式提取追问内容
        pattern = r"\[FQ_START\](.*?)\[FQ_END\]"
        matches = re.findall(pattern, raw_output, re.DOTALL)

        # 处理提取结果
        if matches:
            # 场景1: 三个问题都是完整的，有[FQ_START]和[FQ_END]包裹的完整句子
            cleaned_questions = []
            for q in matches:
                q = q.strip()
                if not q.endswith("？") and not q.endswith("?"):
                    q += "？"
                cleaned_questions.append(q)
            return cleaned_questions
        else:
            # 场景2: 三个问题都不完整，用问号或换行符分割
            # 先移除[FQ_START]和[FQ_END]标记
            clean_output = re.sub(r"\[FQ_START\]|\[FQ_END\]", "", raw_output)

            # 尝试用问号分割
            question_marks_split = [q.strip() + "？" for q in re.split(r"[?？]", clean_output) if q.strip()]

            if question_marks_split:
                return question_marks_split

            # 如果问号分割失败，尝试用换行符分割
            newline_split = [line.strip() for line in clean_output.split('\n') if line.strip()]
            filtered_questions = []
            for line in newline_split:
                # 如果行不是很短，且看起来像问题（不是指导或解释文本）
                if len(line) > 10 and not line.startswith("注：") and not line.startswith("说明："):
                    if not line.endswith("？") and not line.endswith("?"):
                        line += "？"
                    filtered_questions.append(line)

            if filtered_questions:
                return filtered_questions

            # 场景3: 只有一个问题的情况
            # 如果只剩下一个长字符串，尝试识别其中的问题
            if clean_output:
                # 如果整个输出内容看起来像一个问题，将其作为单个问题返回
                clean_output = clean_output.strip()
                if not clean_output.endswith("？") and not clean_output.endswith("?"):
                    clean_output += "？"
                return [clean_output]

            # 如果所有方法都失败，返回默认问题
            return DEFAULT_QUESTIONS

    except re.error as e:
        print(f"正则表达式处理失败: {e}")
        return DEFAULT_QUESTIONS
    except Exception as e:
        print(f"生成追问时出错: {e}")
        return DEFAULT_QUESTIONS