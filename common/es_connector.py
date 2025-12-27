"""
Elasticsearch连接器
提供ES连接和各种搜索功能
"""
from elasticsearch import Elasticsearch
from typing import List, Dict, Optional, Any, Tuple, Union
import requests
import jieba
import traceback
from collections import defaultdict
from common.config import (
    ES_HOST,
    ES_USER,
    ES_PASSWORD,
    DEFAULT_INDEX,
    DEFAULT_TOP_K,
    DEFAULT_VECTOR_WEIGHT,
    DEFAULT_TEXT_WEIGHT,
    DEFAULT_CHUNK_TYPE,
    BGE_API_URL,
    BGE_API_KEYS,
    KB_ID_PAPER,
    KB_ID_CHUNK
)

# 初始化jieba
jieba.initialize()


class ESConnector:
    """Elasticsearch连接器类"""

    def __init__(self):
        """初始化ES连接器"""
        self.es = None
        self.default_index = DEFAULT_INDEX
        self.bge_api_url = BGE_API_URL
        self.bge_api_keys = BGE_API_KEYS
        self.connect()

    def connect(self) -> bool:
        """建立ES连接"""
        try:
            self.es = Elasticsearch(
                ES_HOST,
                basic_auth=(ES_USER, ES_PASSWORD),
                verify_certs=False,
                timeout=30,
                max_retries=3,
                retry_on_timeout=True
            )

            if not self.es.ping():
                print("✗ 连接Elasticsearch失败")
                return False

            info = self.es.info()
            version = info.get('version', {}).get('number', 'unknown')
            print(f"✓ 成功连接到Elasticsearch (版本: {version})")
            return True

        except Exception as e:
            print(f"✗ 连接Elasticsearch时出错: {e}")
            self.es = None
            return False

    def get_vector_embedding(
        self,
        query_input: Union[str, List[str]]
    ) -> Union[List[float], List[List[float]], None]:
        """
        从BGE API获取查询文本的向量嵌入，支持单个字符串或字符串列表输入

        Args:
            query_input: 查询文本，可以是单个字符串或字符串列表

        Returns:
            单个查询返回单个向量，多个查询返回向量列表，失败返回None
        """
        import random
        import numpy as np

        try:
            # 将输入标准化为列表格式
            if isinstance(query_input, str):
                is_single_input = True
                query_texts = [query_input]
            else:
                is_single_input = False
                query_texts = query_input

            # 确保输入不为空
            if not query_texts or all(not text.strip() for text in query_texts):
                print("错误：查询文本为空")
                return None

            # 准备请求载荷
            payload = {
                "model": "BAAI/bge-m3",
                "input": query_texts,
                "encoding_format": "float"
            }

            # 随机选择API令牌
            api_token = random.choice(self.bge_api_keys)

            headers = {
                "Authorization": f"Bearer {api_token}",
                "Content-Type": "application/json"
            }

            # 发送API请求
            response = requests.post(self.bge_api_url, json=payload, headers=headers, timeout=10)

            # 处理响应
            if response.status_code == 200:
                data = response.json()

                if "data" in data and len(data["data"]) > 0:
                    vectors = []

                    for item in data["data"]:
                        if "embedding" in item:
                            vector = item["embedding"]
                            # 归一化向量
                            norm = np.linalg.norm(vector)
                            if norm > 0:
                                vector = [x / norm for x in vector]
                            vectors.append(vector)

                    # 根据输入类型返回相应的结果
                    if is_single_input and vectors:
                        return vectors[0]
                    else:
                        return vectors
                else:
                    print("API返回值中没有找到向量数据")
                    return None
            else:
                print(f"API错误: {response.status_code}, {response.text}")
                return None

        except Exception as e:
            print(f"获取向量嵌入时出错: {e}")
            return None

    def tokenize_with_jieba(self, text: str) -> str:
        """使用jieba分词处理查询文本"""
        try:
            # 使用jieba进行分词
            tokens = jieba.cut(text, cut_all=False)
            # 将分词结果组合成空格分隔的字符串
            tokenized_text = " ".join(tokens)
            return tokenized_text
        except Exception as e:
            print(f"Jieba分词时出错: {e}")
            return text

    def _build_journal_filter(self, journals: List[str]) -> List[Dict]:
        """构建期刊过滤条件，支持keyword字段"""
        if not journals:
            return []

        # 直接使用terms查询处理多个期刊
        journal_filter = {
            "terms": {
                "journal.keyword": journals
            }
        }

        return [journal_filter]

    def _build_author_filter(self, authors: Union[str, List[str]]) -> List[Dict]:
        """优化后的作者过滤方法"""
        if not authors:
            return []

        # 统一处理输入格式
        author_list = [authors] if isinstance(authors, str) else authors
        valid_authors = [a.strip() for a in author_list if a and a.strip()]

        if not valid_authors:
            return []

        # 合并多个作者的查询条件到单个bool/should中
        should_clauses = []
        for author in valid_authors:
            should_clauses.append(
                {"match": {"authors": {"query": author, "operator": "and"}}}
            )

        return [{
            "bool": {
                "should": should_clauses,
                "minimum_should_match": 1
            }
        }]

    def _build_filter_conditions(
        self,
        chunk_type: Optional[List[str]] = None,
        docnm_kwds: Optional[List[str]] = None,
        journals: Optional[Union[str, List[str]]] = None,
        authors: Optional[Union[str, List[str]]] = None,
        year_range: Optional[List[int]] = None,
        language: Optional[str] = None
    ) -> Tuple[List[Dict], Optional[Dict]]:
        """
        构建过滤条件

        Args:
            chunk_type: 文档类型筛选条件
            docnm_kwds: 文档名称关键词筛选条件
            journals: 期刊筛选条件
            authors: 作者筛选条件
            year_range: 年份范围筛选条件
            language: 语言筛选条件

        Returns:
            Tuple[List[Dict], Optional[Dict]]: 返回must过滤条件列表和chunk_type条件
        """
        must_clauses = []
        chunk_type_condition = None

        # 保存用户指定的chunk_type条件
        if chunk_type and len(chunk_type) > 0:
            print(f"保存文档类型筛选: {chunk_type}，将在检索中应用")
            chunk_type_condition = {"terms": {"chunk_type.keyword": chunk_type}}
        else:
            print("未指定文档类型")
            chunk_type_condition = None

        # 添加文档名称关键词筛选
        if docnm_kwds and len(docnm_kwds) > 0:
            print(f"应用文档名称筛选: {docnm_kwds}")
            must_clauses.append({"terms": {"docnm_kwd": docnm_kwds}})

        # 添加期刊过滤条件
        if journals:
            print(f"应用期刊筛选条件: {journals}")
            journal_filters = self._build_journal_filter(journals)
            if len(journal_filters) == 1:
                must_clauses.append(journal_filters[0])
            elif len(journal_filters) > 1:
                must_clauses.append({
                    "bool": {
                        "should": journal_filters,
                        "minimum_should_match": 1
                    }
                })

        # 添加作者过滤条件
        if authors:
            print(f"应用作者筛选条件: {authors}")
            author_filters = self._build_author_filter(authors)
            if len(author_filters) == 1:
                must_clauses.append(author_filters[0])
            elif len(author_filters) > 1:
                must_clauses.append({
                    "bool": {
                        "should": author_filters,
                        "minimum_should_match": 1
                    }
                })

        # 添加年份范围过滤条件
        if year_range and len(year_range) == 2:
            print(f"应用年份范围筛选条件: {year_range}")
            must_clauses.append({
                "range": {
                    "year": {
                        "gte": year_range[0],
                        "lte": year_range[1]
                    }
                }
            })

        # 添加文献语言
        if language:
            print(f"应用文献语言筛选: {language}")
            must_clauses.append({"term": {"language.keyword": language}})

        return must_clauses, chunk_type_condition

    def search_documents(
        self,
        query: str,
        index_name: str = DEFAULT_INDEX,
        top_k: int = DEFAULT_TOP_K,
        vector_weight: float = 0.9,
        text_weight: float = 0.1,
        year_range: Optional[List[int]] = None,
        kb_id: Optional[str] = None,
        chunk_type: Optional[List[str]] = DEFAULT_CHUNK_TYPE,
        docnm_kwds: Optional[List[str]] = None,
        journals: Optional[Union[str, List[str]]] = None,
        authors: Optional[Union[str, List[str]]] = None,
        language: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        对外暴露的搜索函数 - 用于 /api/search_documents

        调用hybrid_search方法执行混合检索
        """
        import time

        total_start = time.time()
        print(f"\n[时间戳] === 开始搜索文档 ===")
        print(f"[时间戳] 查询: '{query}', 知识库ID: {kb_id}, 文档类型: {chunk_type}")

        # 确保连接到ES
        if not self.es:
            if not self.connect():
                return []

        # 调用hybrid_search函数
        results, search_time = self.hybrid_search(
            query_text=query,
            index_name=index_name,
            top_k=top_k,
            vector_weight=vector_weight,
            text_weight=text_weight,
            kb_id=kb_id,
            chunk_type=chunk_type,
            docnm_kwds=docnm_kwds,
            journals=journals,
            authors=authors,
            year_range=year_range,
            language=language
        )

        total_time = time.time() - total_start
        print(f"搜索完成，耗时: {search_time:.2f}秒，总耗时: {total_time:.2f}秒，找到结果数: {len(results)}")
        return results

    def search_literature(
        self,
        query: str,
        index_name: str = DEFAULT_INDEX,
        year_range: Optional[List[int]] = None,
        kb_id: Optional[str] = "3dcd9e360c6811f081000242ac120004",
        docnm_kwds: Optional[List[str]] = None,
        journals: Optional[Union[str, List[str]]] = None,
        authors: Optional[Union[str, List[str]]] = None,
        language: Optional[str] = None,
        levels: Optional[List[str]] = None,
        page: int = 1,
        page_size: int = 10,
        text_weight: float = 1.0
    ) -> Dict[str, Any]:
        """
        对外暴露的搜索函数 - 用于 /api/search_literature

        使用简单的文本检索，支持分页
        """
        import time
        import math

        total_start = time.time()
        print(f"\n[时间戳] === 开始搜索文献 ===")
        print(f"[时间戳] 查询: '{query}', 知识库ID: {kb_id}, 分页信息: 第{page}页，每页{page_size}条")

        # 确保连接到ES
        if not self.es:
            if not self.connect():
                return {
                    "results": [],
                    "total": 0,
                    "page": page,
                    "page_size": page_size,
                    "total_pages": 0,
                    "has_next": False,
                    "has_prev": False
                }

        # 计算from参数
        from_param = (page - 1) * page_size

        # 对查询文本进行分词处理
        tokenized_query = self.tokenize_with_jieba(query)

        # 构建基本查询条件
        must_clauses = []

        # 添加固定的KB_ID过滤
        must_clauses.append({"term": {"kb_id": kb_id}})

        # 添加文档名称关键词筛选
        if docnm_kwds and len(docnm_kwds) > 0:
            must_clauses.append({"terms": {"docnm_kwd": docnm_kwds}})

        # 添加期刊过滤条件
        if journals:
            journal_filters = self._build_journal_filter(journals)
            if len(journal_filters) == 1:
                must_clauses.append(journal_filters[0])
            elif len(journal_filters) > 1:
                must_clauses.append({
                    "bool": {
                        "should": journal_filters,
                        "minimum_should_match": 1
                    }
                })

        # 添加作者过滤条件
        if authors:
            author_filters = self._build_author_filter(authors)
            if len(author_filters) == 1:
                must_clauses.append(author_filters[0])
            elif len(author_filters) > 1:
                must_clauses.append({
                    "bool": {
                        "should": author_filters,
                        "minimum_should_match": 1
                    }
                })

        # 添加年份范围过滤条件
        if year_range and len(year_range) == 2:
            must_clauses.append({
                "range": {
                    "year": {
                        "gte": year_range[0],
                        "lte": year_range[1]
                    }
                }
            })

        # 添加文献语言
        if language:
            must_clauses.append({"term": {"language.keyword": language}})

        # 添加层级过滤
        if levels:
            must_clauses.append({"terms": {"level.keyword": levels}})

        # 构建查询体
        query_body = {
            "query": {
                "bool": {
                    "must": must_clauses,
                    "should": [
                        {"match": {"content_ltks": {"query": tokenized_query, "boost": text_weight * 10}}},
                        {"match_phrase": {"content_ltks": {"query": query, "boost": text_weight * 5}}}
                    ],
                    "minimum_should_match": 1
                }
            },
            "_source": [
                "title", "abstract", "authors", "journal", "year", "vO", "issue",
                "page_range", "pdf_url", "level", "subject", "impact_factor",
                "reference", "docnm_kwd", "translated_abstract", "language", "content_with_weight"
            ],
            "track_scores": True,
            "track_total_hits": True,
            "sort": [{"_score": {"order": "desc"}}],
            "size": page_size,
            "from": from_param
        }

        try:
            # 执行查询
            response = self.es.search(index=index_name, body=query_body)

            hits = response["hits"]["hits"]
            total_hits = response["hits"]["total"]["value"] if isinstance(
                response["hits"]["total"], dict
            ) else response["hits"]["total"]

            # 计算分页信息
            total_pages = math.ceil(total_hits / page_size)
            has_next = page < total_pages
            has_prev = page > 1

            total_time = time.time() - total_start
            print(f"[时间戳] 找到结果总数: {total_hits}，当前页结果数: {len(hits)}, 总耗时: {total_time:.3f}秒")

            return {
                "results": hits,
                "total": total_hits,
                "page": page,
                "page_size": page_size,
                "total_pages": total_pages,
                "has_next": has_next,
                "has_prev": has_prev
            }

        except Exception as e:
            print(f"[时间戳] 搜索出错: {str(e)}")
            return {
                "results": [],
                "total": 0,
                "page": page,
                "page_size": page_size,
                "total_pages": 0,
                "has_next": False,
                "has_prev": False
            }

    def hybrid_search_v4(
        self,
        query_text: str,
        translated_text: Optional[str] = None,
        index_name: str = DEFAULT_INDEX,
        top_k: int = DEFAULT_TOP_K,
        vector_weight: float = DEFAULT_VECTOR_WEIGHT,
        text_weight: float = DEFAULT_TEXT_WEIGHT,
        kb_id: Optional[str] = None,
        chunk_type: Optional[List[str]] = None,
        docnm_kwds: Optional[List[str]] = None,
        journals: Optional[Union[str, List[str]]] = None,
        authors: Optional[Union[str, List[str]]] = None,
        year_range: Optional[List[int]] = None,
        language: Optional[str] = None
    ) -> Tuple[Dict[str, List[Dict[str, Any]]], float]:
        """
        执行多阶段混合检索 - 用于 academic_searcher.py

        第一阶段：在kb_id_paper中检索，获取docnm_kwd和元数据
        第二阶段：多线程处理，获取非raw chunks和raw chunks
        第三阶段：获取相邻chunks并合并内容
        第四阶段：过滤和返回结果

        Returns:
            ({chunks: [], doc_aggs: []}, elapsed_time)
        """
        import time
        import threading
        from queue import Queue

        if not self.es and not self.connect():
            return {"chunks": [], "doc_aggs": []}, 0

        search_start = time.time()
        print(f"[时间戳] 开始混合搜索v4: {time.strftime('%H:%M:%S', time.localtime())}")

        try:
            # ========== 预处理阶段 ==========
            tokenized_query = query_text

            if translated_text:
                tokenized_translated = translated_text

            # ========== 处理过滤条件 ==========
            must_clauses, chunk_type_condition = self._build_filter_conditions(
                chunk_type=chunk_type,
                docnm_kwds=docnm_kwds,
                journals=journals,
                authors=authors,
                year_range=year_range,
                language=language
            )

            # ========== 第一阶段：在KB_ID_PAPER中文本检索 ==========
            filter_clauses = must_clauses.copy()
            filter_clauses.append({"term": {"kb_id": KB_ID_PAPER}})

            should_clauses = [
                {"match": {"text_ltks": {"query": tokenized_query, "boost": text_weight * 10}}}
            ]

            if translated_text and tokenized_translated:
                should_clauses.append(
                    {"match": {"text_ltks": {"query": tokenized_translated, "boost": text_weight * 10}}}
                )

            meta_fields = ["title", "abstract", "authors", "journal", "year", "vO", "issue",
                          "page_range", "pdf_url", "level", "subject", "impact_factor",
                          "reference", "docnm_kwd", "translated_abstract", "language"]

            sm_text_query = {
                "_source": meta_fields,
                "size": 10,
                "query": {
                    "bool": {
                        "filter": filter_clauses,
                        "should": should_clauses,
                        "minimum_should_match": 1
                    }
                }
            }

            sm_text_results = self.es.search(index=index_name, body=sm_text_query, request_timeout=10)
            sm_text_hits = sm_text_results.get("hits", {}).get("hits", [])

            # 提取docnm_kwd列表和保存元数据
            docnm_kwd_list = []
            doc_aggs = []

            for hit in sm_text_hits:
                source = hit.get("_source", {})
                docnm_kwd = source.get("docnm_kwd", "")
                if docnm_kwd:
                    docnm_kwd_list.append(docnm_kwd)
                    meta_data = {k: source.get(k, "") for k in meta_fields}
                    meta_data["id"] = hit["_id"]
                    doc_aggs.append(meta_data)

            docnm_kwd_list = list(set(docnm_kwd_list))

            print(f"第一阶段获取候选文档数: {len(docnm_kwd_list)}")

            if not docnm_kwd_list:
                print("第一阶段检索未找到结果")
                return {"chunks": [], "doc_aggs": []}, time.time() - search_start

            content_fields = ["kb_id", "chunk_type", "content_with_weight", "docnm_kwd"]

            # ========== 第二阶段：多线程处理 ==========
            def extract_fields(source, field_list):
                return {k: source.get(k, "") for k in field_list}

            chunks = []
            raw_chunks = []
            result_queue = Queue()

            # 线程1：获取非raw chunks
            def fetch_non_raw_chunks():
                non_raw_chunks_result = []
                try:
                    msearch_body = []
                    for docnm_kwd in docnm_kwd_list:
                        msearch_body.append({})

                        text_should_clauses = [
                            {"match": {"content_ltks": {"query": tokenized_query, "boost": text_weight * 5}}}
                        ]

                        if translated_text and tokenized_translated:
                            text_should_clauses.append(
                                {"match": {"content_ltks": {"query": tokenized_translated, "boost": text_weight * 1}}}
                            )

                        filter_clauses = [
                            {"term": {"kb_id": KB_ID_CHUNK}},
                            {"term": {"docnm_kwd": docnm_kwd}}
                        ]

                        if chunk_type and len(chunk_type) > 0:
                            filter_clauses.append({"terms": {"chunk_type.keyword": chunk_type}})
                        else:
                            filter_clauses.append({"bool": {"must_not": [{"term": {"chunk_type.keyword": "raw"}}]}})

                        non_raw_query = {
                            "_source": content_fields + ["chunk_id"],
                            "size": 1,
                            "min_score": 0.5,
                            "query": {
                                "bool": {
                                    "filter": filter_clauses,
                                    "should": text_should_clauses,
                                    "minimum_should_match": 1
                                }
                            }
                        }

                        msearch_body.append(non_raw_query)

                    msearch_results = self.es.msearch(body=msearch_body, index=index_name, request_timeout=10)
                    responses = msearch_results.get("responses", [])

                    for i, response in enumerate(responses):
                        if i >= len(docnm_kwd_list):
                            break

                        non_raw_hits = response.get("hits", {}).get("hits", [])
                        for hit in non_raw_hits:
                            source = hit.get("_source", {})
                            chunk_id = source.get("chunk_id")
                            content = extract_fields(source, content_fields)
                            content["id"] = hit["_id"]
                            content["chunk_id"] = chunk_id
                            non_raw_chunks_result.append(content)

                    print(f"获取非raw chunks数量: {len(non_raw_chunks_result)}")
                except Exception as e:
                    print(f"获取非raw chunks异常: {str(e)}")

                result_queue.put(("non_raw", non_raw_chunks_result))

            # 线程2：获取raw chunks
            def fetch_raw_chunks():
                raw_chunks_result = []
                try:
                    msearch_body = []
                    for docnm_kwd in docnm_kwd_list:
                        msearch_body.append({})

                        text_should_clauses = [
                            {"match": {"content_ltks": {"query": tokenized_query, "boost": text_weight * 5}}}
                        ]

                        if translated_text and tokenized_translated:
                            text_should_clauses.append(
                                {"match": {"content_ltks": {"query": tokenized_translated, "boost": text_weight * 5}}}
                            )

                        text_query = {
                            "_source": content_fields + ["chunk_id"],
                            "size": 1,
                            "min_score": 0.5,
                            "query": {
                                "bool": {
                                    "filter": [
                                        {"term": {"kb_id": KB_ID_CHUNK}},
                                        {"term": {"docnm_kwd": docnm_kwd}},
                                        {"term": {"chunk_type.keyword": "raw"}}
                                    ],
                                    "should": text_should_clauses,
                                    "minimum_should_match": 1
                                }
                            }
                        }

                        msearch_body.append(text_query)

                    msearch_results = self.es.msearch(body=msearch_body, index=index_name, request_timeout=10)
                    responses = msearch_results.get("responses", [])

                    for i, response in enumerate(responses):
                        if i >= len(docnm_kwd_list):
                            break

                        raw_hits = response.get("hits", {}).get("hits", [])
                        for raw_hit in raw_hits:
                            raw_source = raw_hit.get("_source", {})
                            chunk_id = raw_source.get("chunk_id")
                            raw_content = extract_fields(raw_source, content_fields)
                            raw_content["id"] = raw_hit["_id"]
                            raw_content["chunk_id"] = chunk_id
                            raw_chunks_result.append(raw_content)

                    print(f"获取raw chunks数量: {len(raw_chunks_result)}")
                except Exception as e:
                    print(f"获取raw chunks异常: {str(e)}")

                result_queue.put(("raw", raw_chunks_result))

            # 创建并启动线程
            thread_non_raw = threading.Thread(target=fetch_non_raw_chunks)
            thread_raw = threading.Thread(target=fetch_raw_chunks)

            thread_non_raw.start()
            thread_raw.start()

            # 等待线程完成
            thread_non_raw.join()
            thread_raw.join()

            # 从队列获取结果
            non_raw_chunks = []
            while not result_queue.empty():
                chunk_type_result, chunk_results = result_queue.get()
                if chunk_type_result == "non_raw":
                    non_raw_chunks = chunk_results
                else:
                    raw_chunks = chunk_results

            # ========== 第三阶段：获取相邻chunks并合并内容 ==========
            raw_chunks_by_docnm = defaultdict(list)
            for raw_chunk in raw_chunks:
                if "chunk_id" not in raw_chunk or raw_chunk["chunk_id"] is None:
                    continue
                docnm_kwd = raw_chunk.get("docnm_kwd", "")
                if docnm_kwd:
                    raw_chunks_by_docnm[docnm_kwd].append(raw_chunk)

            adjacent_chunk_ids_to_fetch = defaultdict(set)
            chunk_id_map = {}

            for docnm_kwd, chunks_in_doc in raw_chunks_by_docnm.items():
                existing_chunk_ids = set()
                for chunk in chunks_in_doc:
                    try:
                        chunk_id = int(chunk.get("chunk_id"))
                        existing_chunk_ids.add(chunk_id)
                        chunk_id_map[(chunk_id, docnm_kwd)] = chunk
                    except (ValueError, TypeError):
                        continue

                for chunk_id in existing_chunk_ids:
                    prev_id = chunk_id - 1
                    next_id = chunk_id + 1

                    if prev_id > 0 and prev_id not in existing_chunk_ids:
                        adjacent_chunk_ids_to_fetch[docnm_kwd].add(prev_id)

                    if next_id not in existing_chunk_ids:
                        adjacent_chunk_ids_to_fetch[docnm_kwd].add(next_id)

            # 批量获取相邻chunks
            adjacent_msearch_body = []
            for docnm_kwd, adj_ids in adjacent_chunk_ids_to_fetch.items():
                adj_ids_list = list(adj_ids)
                for i in range(0, len(adj_ids_list), 50):
                    batch_ids = adj_ids_list[i:i + 50]

                    adjacent_msearch_body.append({})

                    adjacent_query = {
                        "_source": content_fields + ["chunk_id"],
                        "size": len(batch_ids),
                        "query": {
                            "bool": {
                                "must": [
                                    {"term": {"kb_id": KB_ID_CHUNK}},
                                    {"term": {"docnm_kwd": docnm_kwd}},
                                    {"terms": {"chunk_id": [str(id) for id in batch_ids]}}
                                ]
                            }
                        }
                    }
                    adjacent_msearch_body.append(adjacent_query)

            adjacent_chunks_by_key = {}

            if adjacent_msearch_body:
                adjacent_msearch_results = self.es.msearch(
                    body=adjacent_msearch_body,
                    index=index_name,
                    request_timeout=10
                )
                adjacent_responses = adjacent_msearch_results.get("responses", [])

                for response in adjacent_responses:
                    adj_hits = response.get("hits", {}).get("hits", [])

                    for adj_hit in adj_hits:
                        adj_source = adj_hit.get("_source", {})
                        adj_chunk_id = adj_source.get("chunk_id")
                        adj_docnm_kwd = adj_source.get("docnm_kwd", "")

                        if adj_chunk_id is not None and adj_docnm_kwd:
                            try:
                                adj_chunk_id_int = int(adj_chunk_id)
                                adj_content = extract_fields(adj_source, content_fields)
                                adj_content["id"] = adj_hit["_id"]
                                adj_content["chunk_id"] = adj_chunk_id
                                adjacent_chunks_by_key[(adj_chunk_id_int, adj_docnm_kwd)] = adj_content
                            except (ValueError, TypeError):
                                continue

            # 合并相邻内容
            for (chunk_id, docnm_kwd), raw_chunk in chunk_id_map.items():
                original_content = raw_chunk.get("content_with_weight", "")
                merged_content = original_content

                # 添加前一个chunk
                prev_id = chunk_id - 1
                prev_key = (prev_id, docnm_kwd)

                if prev_key in chunk_id_map:
                    prev_content = chunk_id_map[prev_key].get("content_with_weight", "")
                    if prev_content:
                        merged_content = prev_content + "\n\n" + merged_content
                elif prev_key in adjacent_chunks_by_key:
                    prev_content = adjacent_chunks_by_key[prev_key].get("content_with_weight", "")
                    if prev_content:
                        merged_content = prev_content + "\n\n" + merged_content

                # 添加后一个chunk
                next_id = chunk_id + 1
                next_key = (next_id, docnm_kwd)

                if next_key in chunk_id_map:
                    next_content = chunk_id_map[next_key].get("content_with_weight", "")
                    if next_content:
                        merged_content = merged_content + "\n\n" + next_content
                elif next_key in adjacent_chunks_by_key:
                    next_content = adjacent_chunks_by_key[next_key].get("content_with_weight", "")
                    if next_content:
                        merged_content = merged_content + "\n\n" + next_content

                raw_chunk["content_with_weight"] = merged_content

            chunks = non_raw_chunks + raw_chunks

            # ========== 第四阶段：过滤和返回 ==========
            filtered_chunks = []
            for chunk in chunks:
                content = chunk.get("content_with_weight", "")

                if "作者简介" in content:
                    continue

                digit_count = sum(1 for c in content if c.isdigit())
                total_count = len(content) if content else 1
                digit_ratio = digit_count / total_count

                if digit_ratio < 0.3:
                    filtered_chunks.append(chunk)

            # 排序
            def sort_key_for_chunks(chunk):
                docnm_kwd = chunk.get("docnm_kwd", "")
                chunk_id_str = chunk.get("chunk_id", "0")
                try:
                    chunk_id = int(chunk_id_str)
                except (ValueError, TypeError):
                    chunk_id = 0
                return (docnm_kwd, chunk_id)

            filtered_chunks.sort(key=sort_key_for_chunks)

            # 过滤doc_aggs
            remaining_docnm_kwds = set()
            for chunk in filtered_chunks:
                docnm_kwd = chunk.get("docnm_kwd", "")
                if docnm_kwd:
                    remaining_docnm_kwds.add(docnm_kwd)

            filtered_doc_aggs = []
            for doc in doc_aggs:
                doc_kwd = doc.get("docnm_kwd", "")
                if doc_kwd in remaining_docnm_kwds:
                    filtered_doc_aggs.append(doc)

            filtered_doc_aggs.sort(key=lambda doc: doc.get("docnm_kwd", ""))

            result = {
                "chunks": filtered_chunks,
                "doc_aggs": filtered_doc_aggs
            }

            elapsed_time = time.time() - search_start
            print(f"混合检索完成: chunks数: {len(filtered_chunks)}, doc_aggs数: {len(filtered_doc_aggs)}, 总耗时: {elapsed_time:.2f}s")

            return result, elapsed_time

        except Exception as e:
            print(f"搜索异常: {str(e)}")
            traceback.print_exc()
            return {"chunks": [], "doc_aggs": []}, time.time() - search_start

    def search_documents_with_vector(
            self,
            query: str,
            index_name: str = DEFAULT_INDEX,
            top_k: int = DEFAULT_TOP_K,
            year_range: Optional[List[int]] = None,
            kb_id: Optional[str] = None,
            chunk_type: Optional[List[str]] = DEFAULT_CHUNK_TYPE,
            docnm_kwds: Optional[List[str]] = None,
            journals: Optional[Union[str, List[str]]] = None,
            authors: Optional[Union[str, List[str]]] = None,
            language: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        对外暴露的搜索函数，使用纯向量检索

        Args:
            query: 查询文本
            index_name: 索引名称
            top_k: 返回结果数量
            year_range: 年份范围列表 [from_year, to_year]
            kb_id: 知识库ID过滤
            chunk_type: 文档类型过滤
            docnm_kwds: 文章列表，格式为[docnm_kwd1, docnm_kwd2, ...]
            journals: 期刊名称，字符串或字符串列表
            authors: 作者名称，字符串或字符串列表
            language: 文献语言

        Returns:
            检索结果列表
        """
        import time

        total_start = time.time()
        print(f"\n[时间戳] === 开始搜索文档（纯向量） ===")
        print(f"[时间戳] 查询: '{query}', 知识库ID: {kb_id}, 文档类型: {chunk_type}")
        if journals:
            print(f"[时间戳] 期刊筛选: {journals}")
        if authors:
            print(f"[时间戳] 作者筛选: {authors}")
        if year_range:
            print(f"[时间戳] 年份范围: {year_range}")
        if docnm_kwds:
            print(f"[时间戳] 文档名称: {docnm_kwds}")
        if language:
            print(f"[时间戳] 文献语言: {language}")

        # 确保连接到ES
        if not self.es:
            es_connect_start = time.time()
            if not self.connect():
                total_end = time.time()
                print(f"[时间戳] 搜索总耗时(连接失败): {(total_end - total_start):.3f}秒")
                return []
            es_connect_end = time.time()
            print(f"[时间戳] ES连接函数调用耗时: {(es_connect_end - es_connect_start):.3f}秒")

        # 调用vector_search函数
        results, search_time = self.vector_search(
            query_text=query,
            index_name=index_name,
            top_k=top_k,
            kb_id=kb_id,
            chunk_type=chunk_type,
            docnm_kwds=docnm_kwds,
            journals=journals,
            authors=authors,
            year_range=year_range,
            language=language
        )

        total_end = time.time()
        total_time = total_end - total_start
        print(f"搜索完成，耗时: {search_time:.2f}秒，总耗时: {total_time:.2f}秒，找到结果数: {len(results)}")
        return results

    def vector_search(
            self,
            query_text: str,
            index_name: str = DEFAULT_INDEX,
            top_k: int = DEFAULT_TOP_K,
            kb_id: Optional[str] = None,
            chunk_type: Optional[List[str]] = None,
            docnm_kwds: Optional[List[str]] = None,
            journals: Optional[Union[str, List[str]]] = None,
            authors: Optional[Union[str, List[str]]] = None,
            year_range: Optional[List[int]] = None,
            language: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        执行纯向量检索

        Args:
            query_text: 查询文本
            index_name: 索引名称
            top_k: 返回结果数量
            kb_id: 知识库ID
            chunk_type: 文档类型列表
            docnm_kwds: 文档名称关键词列表
            journals: 期刊列表
            authors: 作者列表
            year_range: 年份范围
            language: 语言

        Returns:
            (搜索结果列表, 搜索耗时)
        """
        import time

        search_start = time.time()

        if not self.es and not self.connect():
            return [], 0

        try:
            # 获取查询向量
            query_vector = self.get_vector_embedding(query_text)

            if not query_vector:
                print("向量生成失败")
                return [], time.time() - search_start

            # 构建过滤条件
            must_clauses, chunk_type_condition = self._build_filter_conditions(
                chunk_type=chunk_type,
                docnm_kwds=docnm_kwds,
                journals=journals,
                authors=authors,
                year_range=year_range,
                language=language
            )

            # 添加kb_id过滤
            if kb_id:
                must_clauses.append({"term": {"kb_id": kb_id}})

            # 添加chunk_type条件（如果存在）
            if chunk_type_condition:
                must_clauses.append(chunk_type_condition)

            # 构建KNN查询
            knn_query = {
                "_source": [
                    "title", "abstract", "authors", "journal", "year", "vO", "issue",
                    "page_range", "doc_id", "kb_id", "chunk_type", "content_with_weight",
                    "pdf_url", "level", "subject", "impact_factor", "reference",
                    "docnm_kwd", "translated_abstract", "language"
                ],
                "size": top_k,
                "knn": {
                    "field": "q_1024_vec",
                    "query_vector": query_vector,
                    "k": top_k,
                    "num_candidates": min(top_k * 3, 100)
                }
            }

            # 如果有过滤条件，添加到knn查询中
            if must_clauses:
                knn_query["knn"]["filter"] = {
                    "bool": {
                        "must": must_clauses
                    }
                }

            # 执行查询
            response = self.es.search(index=index_name, body=knn_query, request_timeout=30)

            hits = response.get("hits", {}).get("hits", [])

            # 格式化结果
            results = []
            for hit in hits:
                source = hit.get("_source", {})
                result = {
                    "id": hit["_id"],
                    "score": hit.get("_score", 0),
                    **{k: source.get(k, "") for k in [
                        'title', 'abstract', 'authors', 'journal', 'year', 'vO',
                        'issue', 'page_range', 'doc_id', 'kb_id', 'chunk_type',
                        'content_with_weight', 'pdf_url', 'level', 'subject',
                        'impact_factor', 'reference', 'docnm_kwd', 'translated_abstract',
                        'language'
                    ]}
                }
                results.append(result)

            elapsed_time = time.time() - search_start
            print(f"向量检索完成，找到 {len(results)} 个结果，耗时: {elapsed_time:.2f}秒")

            return results, elapsed_time

        except Exception as e:
            print(f"向量检索异常: {str(e)}")
            import traceback
            traceback.print_exc()
            return [], time.time() - search_start