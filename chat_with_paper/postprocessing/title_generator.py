"""
标题生成器
基于对话内容生成简洁明了的标题
"""
import re
from typing import Dict
from common.llm_call import LLMAPIHandler
from common.config import OPENAI_API_KEY

# 初始化LLM处理器
llm_handler = LLMAPIHandler(openai_api_key=OPENAI_API_KEY)


def generate_dialogue_title(dialogue_dict: Dict[str, str]) -> str:
    """
    基于对话字典生成简洁明了的对话标题

    Args:
        dialogue_dict: {
            "user": "合并后的所有用户发言内容",
            "assistant": "合并后的所有助手回复内容"
        }

    Returns:
        代表对话主题的标题字符串
    """
    # 输入验证
    if not isinstance(dialogue_dict, dict):
        print("错误: 输入必须是字典类型")
        return "未分类对话主题"
    if "user" not in dialogue_dict or "assistant" not in dialogue_dict:
        print("错误: 字典必须包含'user'和'assistant'键")
        return "未分类对话主题"

    combined_text = dialogue_dict["user"] + " " + dialogue_dict["assistant"]

    # 默认标题
    DEFAULT_TITLE = combined_text[:10] + "……"

    try:
        # 构建对话历史字符串
        history_dialogue = f"用户: {dialogue_dict['user']}\n助手: {dialogue_dict['assistant']}"

        prompt = f"""<TASK>
你是一个对话标题生成引擎，目标是为对话生成简洁明了且具有代表性的标题。请基于<HISTORY_DIALOGUE>中的对话内容，生成一个合适的标题，严格遵循<TITLE_GUIDELINES>中的规则。
</TASK>

<HISTORY_DIALOGUE>
{history_dialogue}
</HISTORY_DIALOGUE>

<TITLE_GUIDELINES>
## 标题生成准则
1. 核心主题提取策略：
   - 关键词提取：识别对话中高频出现的专业术语或核心概念
   - 主旨概括：捕捉对话的核心问题或讨论重点
   - 领域归类：确定对话所属的专业或知识领域

2. 标题生成原则：
   - 长度控制在5-40个汉字之间
   - 避免使用过于笼统的词语（如"讨论"、"探讨"等）
   - 使用名词性短语结构，避免使用完整句子
   - 不使用标点符号（包括问号、感叹号等）

3. 安全过滤机制：
   - 禁止生成以下类型的标题：
     * 包含政治敏感内容的标题
     * 涉及特定政策评价的标题
     * 针对特定群体的歧视性标题
     * 涉及制度批评的标题
     * 有明显意识形态倾向的标题
     * 涉及宗教信仰判断的标题
     * 暗示地缘政治立场的标题
     * 含有民族或区域偏见的标题
     * 讨论社会争议敏感事件的标题
   - 发现敏感内容时，应生成中性的学术性标题
</TITLE_GUIDELINES>

<OUTPUT_INSTRUCTION>
1. 标题必须用[TITLE_START]和[TITLE_END]严格包裹
2. 标题应为中文名词性短语，不使用标点符号
3. 标题长度控制在40个汉字以内
4. 不要添加任何解释或其他内容
</OUTPUT_INSTRUCTION>

标题："""

        try:
            # 调用LLM API生成标题
            response = llm_handler.call_llm(
                provider="openai",
                prompt=prompt,
                model="gpt-4o-mini",
                max_tokens=100,
                temperature=0.7
            )

            raw_output = response.strip()

            # 提取标题
            pattern = r"\[TITLE_START\](.*?)\[TITLE_END\]"
            matches = re.findall(pattern, raw_output, re.DOTALL)

            if matches:
                title = matches[0].strip()
                # 移除所有标点符号并截断长度
                title = re.sub(r'[^\w\s]', '', title)
                if len(title) > 40:
                    title = title[:40]
                return title
            else:
                # 未找到标记包裹的标题，使用默认标题
                raise Exception("未找到标记包裹的标题")

        except Exception as e:
            print(f"API调用或标题提取出错: {str(e)}")
            print("直接截取文字...")
            return DEFAULT_TITLE

    except Exception as e:
        print(f"生成标题出错: {str(e)}")
        return DEFAULT_TITLE