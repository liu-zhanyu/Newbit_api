"""
指标搜索器
在Elasticsearch中搜索相关的数据指标
"""
from typing import List, Dict, Any, Optional
from common.es_connector import ESConnector
from common.config import DATA_INDICATORS_KB_ID

# 初始化ES连接器
es_service = ESConnector()


def search_indicators(
        query: str,
        top_k: int = 3,
        vector_weight: float = 0.7,
        text_weight: float = 0.3
) -> List[Dict[str, Any]]:
    """
    在ES中搜索数据指标

    Args:
        query: 搜索查询（变量名称或描述）
        top_k: 返回结果数量
        vector_weight: 向量搜索权重
        text_weight: 文本搜索权重

    Returns:
        指标搜索结果列表，每个结果包含：
        - docnm_kwd: 指标名称
        - content_with_weight: 指标描述
        - Statistical unit: 统计单位
        - Source: 数据来源

    Example:
        >>> results = search_indicators("教育水平", top_k=3)
        >>> for result in results:
        ...     print(result['docnm_kwd'])
        受教育年限
        最高学历
        在校生人数
    """
    if not es_service.es:
        print("ES连接未建立")
        return []

    try:
        # 使用hybrid_search_indicator方法搜索指标
        # 添加kb_id过滤条件
        filters = {
            "kb_id": DATA_INDICATORS_KB_ID
        }

        results = es_service.hybrid_search_indicator(
            query_text=query,
            top_k=top_k,
            vector_weight=vector_weight,
            text_weight=text_weight,
            filters=filters
        )

        print(f"搜索指标 '{query}': 找到 {len(results)} 个结果")

        return results

    except Exception as e:
        print(f"搜索指标时出错: {e}")
        return []


def search_multiple_indicators(
        queries: List[str],
        top_k_per_query: int = 3
) -> Dict[str, List[Dict[str, Any]]]:
    """
    批量搜索多个指标

    Args:
        queries: 查询列表（变量名称列表）
        top_k_per_query: 每个查询返回的结果数量

    Returns:
        查询到结果列表的映射字典

    Example:
        >>> queries = ["教育水平", "收入", "健康状况"]
        >>> results = search_multiple_indicators(queries, top_k_per_query=3)
        >>> print(results.keys())
        dict_keys(['教育水平', '收入', '健康状况'])
    """
    results_dict = {}

    for query in queries:
        results = search_indicators(query, top_k=top_k_per_query)
        results_dict[query] = results

    return results_dict


def extract_indicator_names(search_results: List[Dict[str, Any]]) -> List[str]:
    """
    从搜索结果中提取指标名称

    Args:
        search_results: 搜索结果列表

    Returns:
        指标名称列表
    """
    indicator_names = []

    for result in search_results:
        # 从docnm_kwd中提取指标名称（移除.txt后缀）
        docnm_kwd = result.get("docnm_kwd", "")
        if docnm_kwd.endswith(".txt"):
            docnm_kwd = docnm_kwd[:-4]

        if docnm_kwd:
            indicator_names.append(docnm_kwd)

    return indicator_names