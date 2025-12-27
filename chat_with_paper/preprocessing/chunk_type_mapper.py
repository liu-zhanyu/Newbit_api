"""
Chunk类型映射器
根据问题类型返回相关的chunk类型
"""
from typing import List


def get_chunk_types(query_type: int) -> List[str]:
    """
    根据问题类型返回相关的chunk类型列表

    Args:
        query_type: 问题类型(0-4)
            0: 行业现状问题
            1: 理论相关问题
            2: 方法相关问题
            3: 科学事实相关问题
            4: 研究进展相关问题

    Returns:
        相关的chunk类型列表，每个类型都是小写且后缀为_answer
    """
    query_type_names = {
        0: "行业现状问题",
        1: "理论相关问题",
        2: "方法相关问题",
        3: "科学事实相关问题",
        4: "研究进展相关问题"
    }
    print(f"问题类型: {query_type} - {query_type_names.get(query_type, '未知类型')}")

    chunk_types_mapping = {
        0: ["background_answer", "implication_answer"],
        1: ["theory_answer", "hypothesis_answer"],
        2: ["method_answer", "data_answer", "sample_answer", "measurement_answer", "analysis_answer"],
        3: ["concept_answer", "conclusion_answer"],
        4: ["question_answer", "contribution_answer", "limitation_answer"]
    }

    if query_type in chunk_types_mapping:
        return chunk_types_mapping[query_type] + ["summary"]
    else:
        raise ValueError(f"无效的问题类型: {query_type}，请输入0-4之间的整数")