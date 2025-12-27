"""
学术参数增强搜索器
基于提取的学术参数进行多阶段精确检索
"""
from typing import List, Dict, Optional, Tuple
from common.es_connector import ESConnector
from common.config import KB_ID_PAPER, DEFAULT_TOP_K
from chat_with_paper.preprocessing.academic_param_extractor import AcademicParamExtractor
from chat_with_paper.preprocessing.query_optimizer import get_optimal_query
from chat_with_paper.preprocessing.query_translator import get_translated_optimal_query

# 初始化组件
es_service = ESConnector()
param_extractor = AcademicParamExtractor()


def search_with_extracted_params_v3(
        es_service: ESConnector,
        user_query: str,
        top_k: int = DEFAULT_TOP_K,
        additional_filters: Optional[Dict] = None
) -> Tuple[Dict, Dict]:
    """
    基于提取的学术参数进行多阶段检索

    工作流程：
    1. 从用户查询中提取学术参数（年份、期刊、作者、语言）
    2. 生成优化后的检索词（同语言）
    3. 生成翻译后的检索词（跨语言）
    4. 执行三阶段混合检索：
       - 阶段1: 使用KB_ID_PAPER获取初步文档
       - 阶段2: 并行获取非raw chunks和raw chunks
       - 阶段3: 获取相邻chunks并合并内容

    Args:
        es_service: Elasticsearch服务实例
        user_query: 用户原始查询
        top_k: 返回结果数量
        additional_filters: 额外的过滤条件（如kb_id、docnm_kwds）

    Returns:
        Tuple[Dict, Dict]:
            - results: 包含chunks和doc_aggs的搜索结果
            - extracted_info: 提取的学术参数信息

    Example:
        results, params = search_with_extracted_params_v3(
            es_service=es_service,
            user_query="近五年关于深度学习的顶刊研究",
            top_k=30
        )
        # params可能包含:
        # {
        #     "year_range": [2020, 2025],
        #     "journals": ["Nature", "Science"],
        #     "language": "中文",
        #     "optimal_query": "深度学习 神经网络",
        #     "translated_query": "deep learning neural network"
        # }
    """
    # 1. 提取学术参数
    extracted_info = param_extractor.extract_all_params(user_query)

    # 2. 生成优化查询（同语言）
    optimal_query = get_optimal_query(user_query)
    extracted_info["optimal_query"] = optimal_query

    # 3. 生成翻译查询（跨语言）
    translated_query = get_translated_optimal_query(user_query)
    extracted_info["translated_query"] = translated_query

    # 4. 构建过滤条件
    filters = additional_filters or {}

    if extracted_info.get("year_range"):
        filters["year_range"] = extracted_info["year_range"]

    if extracted_info.get("journals"):
        filters["journals"] = extracted_info["journals"]

    if extracted_info.get("authors"):
        filters["authors"] = extracted_info["authors"]

    if extracted_info.get("language"):
        filters["language"] = extracted_info["language"]

    # 5. 执行混合搜索（使用优化后的查询）
    # 优先使用优化查询，如果为空则使用原始查询
    search_query = optimal_query if optimal_query else user_query

    # 同时使用中英文查询（如果有翻译）
    if translated_query:
        search_query = f"{search_query} {translated_query}"

    results = es_service.hybrid_search_v4(
        query=search_query,
        top_k=top_k,
        additional_filters=filters
    )

    # 6. 格式化返回结果
    formatted_results = {
        "chunks": results.get("chunks", []),
        "doc_aggs": results.get("doc_aggs", [])
    }

    return formatted_results, extracted_info