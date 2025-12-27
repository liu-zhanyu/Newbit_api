"""
简单文档搜索器
提供基础的文档检索功能
"""
from typing import List, Optional, Dict, Any
from common.es_connector import ESConnector
from common.config import (
    KB_ID_PAPER,
    DEFAULT_TOP_K,
    DEFAULT_VECTOR_WEIGHT,
    DEFAULT_TEXT_WEIGHT
)

# 初始化ES连接器
es_service = ESConnector()


def search_documents_simple(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    docnm_kwds: Optional[List[str]] = None,
    vector_weight: float = DEFAULT_VECTOR_WEIGHT,
    text_weight: float = DEFAULT_TEXT_WEIGHT
) -> List[Dict[str, Any]]:
    """
    简单文档搜索函数 - 使用混合检索（向量+文本）

    Args:
        query: 搜索查询文本
        top_k: 返回结果数量
        docnm_kwds: 文献名称关键词列表（可选），用于筛选特定文献
        vector_weight: 向量搜索权重（默认0.7）
        text_weight: 文本搜索权重（默认0.3）

    Returns:
        搜索结果列表，每个结果包含文档的基本信息和内容

    Example:
        >>> results = search_documents_simple(
        ...     query="深度学习在医疗影像中的应用",
        ...     top_k=30,
        ...     docnm_kwds=["neural network", "medical imaging"]
        ... )
        >>> print(f"找到 {len(results)} 个结果")
    """
    # 调用ES服务的search_documents_with_vector方法（混合检索：向量+文本）
    results = es_service.search_documents_with_vector(
        query=query,
        top_k=top_k,
        vector_weight=vector_weight,
        text_weight=text_weight,
        kb_id=KB_ID_PAPER,  # 固定的知识库ID
        docnm_kwds=docnm_kwds
    )

    return results