"""
摘要总结器
基于多篇研究摘要生成总结，突出核心观点并标注引用
"""
from typing import List, Dict
from common.llm_call import LLMAPIHandler
from common.config import OPENAI_API_KEY

# 初始化LLM处理器
llm_handler = LLMAPIHandler(openai_api_key=OPENAI_API_KEY)


def summarize_abstracts(abstracts: List[Dict[str, str]]) -> str:
    """
    基于多篇研究摘要生成总结

    Args:
        abstracts: 研究摘要列表，格式为:
            [
                {
                    "author": "作者名称",
                    "year": "发表年份",
                    "abstract": "摘要内容"
                },
                ...
            ]

    Returns:
        总结内容字符串，包含引用标注（作者, 年份）

    Raises:
        ValueError: 当输入数据格式不正确时

    Example:
        >>> abstracts = [
        ...     {
        ...         "author": "Zhang et al.",
        ...         "year": "2023",
        ...         "abstract": "This study explores deep learning applications..."
        ...     },
        ...     {
        ...         "author": "Li et al.",
        ...         "year": "2024",
        ...         "abstract": "We propose a novel approach to NLP..."
        ...     }
        ... ]
        >>> summary = summarize_abstracts(abstracts)
        >>> print(summary)
        # 输出示例：
        # 深度学习在自然语言处理中的应用呈现多样化趋势。Zhang et al. (2023)探讨了...
        # Li et al. (2024)提出了新的方法...
    """
    # 输入验证
    if not abstracts or not isinstance(abstracts, list):
        raise ValueError("摘要列表不能为空且必须是数组格式")

    for item in abstracts:
        if not isinstance(item, dict):
            raise ValueError("摘要项必须是字典格式")
        if "abstract" not in item or not item["abstract"]:
            raise ValueError("摘要内容不能为空")
        if "authors" not in item or not item["authors"]:
            raise ValueError("作者信息不能为空")
        if "year" not in item or not item["year"]:
            raise ValueError("年份信息不能为空")

    # 默认返回值（用于错误回退）
    DEFAULT_SUMMARY = "暂无摘要总结"

    try:
        # 构建摘要文本
        abstracts_text = ""
        for idx, item in enumerate(abstracts, 1):
            author = item.get("authors", "Unknown")
            year = item.get("year", "Unknown")
            abstract = item.get("abstract", "")

            abstracts_text += f"\n\n【文献{idx}】\n"
            abstracts_text += f"作者: {author}\n"
            abstracts_text += f"年份: {year}\n"
            abstracts_text += f"摘要: {abstract}\n"

        # 构建prompt
        prompt = f"""<TASK>
你是一个学术文献分析专家，需要对多篇研究摘要进行综合总结。请仔细阅读<ABSTRACTS>中的所有摘要，遵循<GUIDELINES>中的规则生成一个全面的总结。
</TASK>

<ABSTRACTS>
{abstracts_text}
</ABSTRACTS>

<GUIDELINES>
## 总结准则

1. **核心观点提取**：
   - 识别每篇研究的主要发现和核心论点
   - 提炼共同的研究主题和关注点
   - 突出不同研究之间的联系与差异

2. **引用标注规范**：
   - 在陈述具体研究发现时，必须标注来源
   - 引用格式：(作者, 年份)
   - 示例：Zhang et al. (2023)发现...
   - 或：有研究表明...（Li et al., 2024）

3. **内容组织结构**：
   - 按主题或研究方向组织内容，而非逐篇罗列
   - 先概述整体研究趋势，再具体阐述各研究发现
   - 突出研究之间的互补性或矛盾之处

4. **语言要求**：
   - 使用学术性语言，保持客观中立
   - 避免过度简化或夸大研究结论
   - 确保逻辑连贯，过渡自然
   - 长度控制在200-400字

5. **质量标准**：
   - 准确反映原摘要的核心内容
   - 避免遗漏重要发现
   - 不添加摘要中未提及的信息
   - 平衡各篇文献的呈现比重

## 输出格式
直接输出总结内容，不需要标题或其他说明文字。

## 优质示例

输入摘要：
【文献1】
作者: Zhang et al.
年份: 2023
摘要: 本研究探讨了深度学习在图像识别中的应用，提出了一种新的卷积神经网络架构...

【文献2】
作者: Li et al.
年份: 2024
摘要: 我们研究了注意力机制在自然语言处理中的作用，发现多头注意力可以显著提升...

输出总结：
近年来，深度学习技术在多个领域取得了显著进展。在计算机视觉领域，Zhang et al. (2023)提出了一种改进的卷积神经网络架构，通过引入新的池化策略提升了图像识别准确率。与此同时，自然语言处理领域也出现了重要突破。Li et al. (2024)的研究表明，多头注意力机制能够有效捕捉文本中的长距离依赖关系，在机器翻译任务中表现出色。这些研究共同揭示了深度学习架构创新对提升模型性能的重要性，为未来研究提供了有价值的方向。
</GUIDELINES>

<OUTPUT_INSTRUCTION>
请基于以上所有摘要，生成一个综合性的学术总结。记住：
1. 必须标注引用来源（作者, 年份）
2. 按主题组织内容，不要逐篇罗列
3. 200-400字之间
4. 直接输出总结内容，不要任何前言或后语
</OUTPUT_INSTRUCTION>

总结："""

        # 调用LLM API生成总结
        summary = llm_handler.call_llm(
            provider="openai",
            prompt=prompt,
            model="gpt-4o-mini",
            max_tokens=600,
            temperature=0.7
        )

        # 处理返回结果
        if summary and summary.strip():
            # 移除可能的多余空白
            summary = summary.strip()

            # 验证总结是否包含引用
            has_citation = False
            for item in abstracts:
                author = item.get("authors", "")
                year = str(item.get("year", ""))

                # 检查是否包含 "作者 (年份)" 或 "(作者, 年份)" 格式的引用
                if author and year:
                    # 简化的引用检查（可以根据需要增强）
                    if author in summary or year in summary:
                        has_citation = True
                        break

            if not has_citation:
                print("警告：生成的总结可能缺少引用标注")

            return summary
        else:
            print("LLM返回空结果")
            return DEFAULT_SUMMARY

    except ValueError as e:
        # 输入验证错误，直接抛出
        raise

    except Exception as e:
        print(f"生成摘要总结时出错: {e}")
        return DEFAULT_SUMMARY