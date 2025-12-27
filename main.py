"""
FastAPI主应用
提供文献对话和数据对话的API接口
"""
import sys
import os

# 将项目根目录添加到 Python 路径
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import uvicorn
import uuid
import logging
import io
import json
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

# 导入配置

# 导入公共组件
from common.llm_call import LLMAPIHandler
from common.mongodb_connector import MongoDBConnector
from common.es_connector import ESConnector
from common.oss_handler import OSSHandler
from common.state_checker import get_task_state_service

# 导入文献对话模块
from chat_with_paper.preprocessing.query_classifier import classify_query_type
from chat_with_paper.preprocessing.query_rewriter import rewrite_and_disambiguate_query
from chat_with_paper.deep_chat.document_searcher import search_documents_simple
from chat_with_paper.retrieval.academic_searcher import search_with_extracted_params_v3
from chat_with_paper.deep_chat.abstract_summarizer import summarize_abstracts
from chat_with_paper.postprocessing.followup_generator import generate_followup_questions
from chat_with_paper.postprocessing.title_generator import generate_dialogue_title

# 导入数据对话模块
from chat_with_data.retrieval.data_retriever import ResearchDataRetriever
from chat_with_data.deep_chat.data_summarizer import summarize_data
from chat_with_data.deep_chat.data_analyzer import ChatWithData
from chat_with_data.postprocessing.research_data_exporter import async_generate_research_data

# 导入数据模型
from model import (
    ClassifyRequest,
    SearchRequest,
    FollowupRequest,
    SearchDocumentsRequest,
    SearchLiteratureRequest,
    QueryRewriteRequest,
    AbstractsSummarizeRequest,
    DialogueRequest,
    ReviewRequest,
    ChatWithDataRequest,
    ChatRequest,
    StatusRequest,
    MetricsAnalysisRequest
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('literature_review_service.log')
    ]
)
logger = logging.getLogger("literature_review_service")

# 创建FastAPI应用
app = FastAPI(
    title="Newbit服务",
    description="Newbit产品API接口",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化全局服务实例
llm_handler = LLMAPIHandler()
mongo_connector = MongoDBConnector()
es_service = ESConnector()
oss_handler = OSSHandler()
data_retriever = ResearchDataRetriever()


# ==================== 健康检查 ====================

@app.get("/")
def read_root():
    """根路径"""
    logger.info("访问根路径")
    return {"message": "Newbit服务已启动，请访问 /docs 查看API文档"}


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "mongodb": mongo_connector.db is not None,
            "elasticsearch": es_service.es is not None,
            "oss": oss_handler.bucket is not None
        }
    }


# ==================== 文献对话API ====================

@app.post("/api/classify_query", summary="分类用户查询类型")
def classify_query_api(request: ClassifyRequest):
    """
    使用大模型判断用户查询类型

    Args:
        request: 包含用户ID和查询内容的请求

    Returns:
        分类结果（0: QA型，1: 数据型，2: 混合型）
    """
    user_id = request.user_id
    query = request.query

    logger.info(f"API - 用户 {user_id} 发起查询分类，查询内容: {query}")

    try:
        result = classify_query_type(query)
        logger.info(f"API - 用户 {user_id} 查询分类结果: {result}")
        return {"status": "success", "result": result}
    except Exception as e:
        logger.error(f"API - 用户 {user_id} 查询分类失败: {e}")
        raise HTTPException(status_code=500, detail=f"查询分类失败: {e}")


@app.post("/api/rewrite_query", summary="重写并消除查询歧义")
def rewrite_query_api(request: QueryRewriteRequest):
    """
    使用大模型对用户查询进行改写和消除歧义

    Args:
        user_id: 用户ID
        history: 历史对话记录，格式为[{"content": "文本内容", "role": "user/assistant"}]
        query: 当前用户查询内容

    Returns:
        {
            "status": "success/fail",
            "original_query": "原始查询",
            "rewritten_query": "改写后的查询"
        }
    """
    user_id = request.user_id
    history = request.history
    query = request.query

    logger.info(f"API - 用户 {user_id} 发起查询改写，原始查询: '{query}'，历史消息数: {len(history)}")

    try:
        rewritten_query = rewrite_and_disambiguate_query(history, query)

        if rewritten_query != query:
            logger.info(f"API - 用户 {user_id} 查询改写成功: '{rewritten_query}'")
            return {
                "status": "success",
                "original_query": query,
                "rewritten_query": rewritten_query
            }
        else:
            logger.info(f"API - 用户 {user_id} 查询未改写，返回原查询")
            return {
                "status": "success",
                "original_query": query,
                "rewritten_query": query,
                "message": "查询无需改写或改写失败，返回原始查询"
            }

    except ValueError as e:
        logger.error(f"API - 用户 {user_id} 输入验证失败: {e}")
        return {
            "status": "fail",
            "original_query": query,
            "rewritten_query": query,
            "message": f"输入验证失败: {str(e)}"
        }

    except Exception as e:
        logger.critical(f"API - 用户 {user_id} 查询改写异常: {e}", exc_info=True)
        return {
            "status": "fail",
            "original_query": query,
            "rewritten_query": query,
            "message": f"查询改写失败: {str(e)}"
        }


@app.post("/api/search", summary="搜索相关文档")
def api_search_documents(request: SearchRequest):
    """
    简单文档搜索API

    Args:
        request: 包含查询文本和期望结果数量的请求

    Returns:
        {
            "results": {
                "chunks": [chunk信息],
                "doc_aggs": [文档聚合信息]
            },
            "count": 结果数量
        }
    """
    data = search_documents_simple(
        query=request.query,
        top_k=request.top_k,
        docnm_kwds=request.docnm_kwds
    )

    results = {
        "chunks": [
            {k: v for k, v in item.items() if k in ['content_with_weight', 'chunk_type', 'kb_id', 'score']}
            for item in data
        ],
        "doc_aggs": list({
                             item.get("docnm_kwd", item["title"]): {
                                 k: v for k, v in item.items()
                                 if k not in ['content_with_weight', 'chunk_type', 'kb_id', 'score']
                             }
                             for item in data
                         }.values())
    }

    return {"results": results, "count": len(results["chunks"])}


@app.post("/api/search_with_router_v3", summary="结合路由搜索相关文档V3")
def search_with_router_v3(request: SearchRequest):
    """
    结合学术参数提取的高级检索API

    Args:
        request: 包含查询文本和期望结果数量的请求

    Returns:
        {
            "results": {
                "chunks": [chunk信息],
                "doc_aggs": [文档聚合信息]
            },
            "count": 结果数量
        }
    """
    results, extracted_info = search_with_extracted_params_v3(
        es_service=es_service,
        user_query=request.query,
        top_k=request.top_k,
        additional_filters={
            "kb_id": "7750e714049611f08aa20242ac120003",
            "docnm_kwds": request.docnm_kwds
        }
    )

    print("提取的学术参数：", extracted_info)

    # 替换PDF URL
    for doc_agg in results["doc_aggs"]:
        doc_agg["pdf_url"] = doc_agg["pdf_url"].replace(
            "http://hentre-admin-upload.oss-cn-qingdao.aliyuncs.com/",
            "https://oss.gtpa.cloud/"
        )

    return {"results": results, "count": len(results["chunks"])}


@app.post("/api/search/documents", summary="智能搜索文献")
def search_documents_api(request: SearchDocumentsRequest):
    """
    高级文档搜索API，支持多种过滤条件

    Args:
        request: 包含搜索查询和各种过滤条件的请求对象

    Returns:
        {
            "results": {
                "chunks": [chunk信息],
                "doc_aggs": [文档信息]
            },
            "count": 结果数量
        }
    """
    raw_results = es_service.search_documents(
        query=request.query,
        top_k=request.top_k,
        year_range=request.year_range,
        kb_id=request.kb_id,
        journals=request.journals,
        authors=request.authors,
        language=request.language,
        chunk_type=["summary"]
    )

    results = {
        "chunks": [{
            "content_with_weight": item.get("content_with_weight", item.get("content", "")),
            "chunk_type": item.get("chunk_type", ""),
            "kb_id": item.get("kb_id", ""),
            "score": item.get("score", 0.0)
        } for item in raw_results],

        "doc_aggs": list({
                             item.get("docnm_kwd", item.get("title", "")): {
                                 "title": item.get("title", ""),
                                 "abstract": item.get("abstract", ""),
                                 "authors": item.get("authors", ""),
                                 "journal": item.get("journal", ""),
                                 "year": item.get("year", ""),
                                 "vO": item.get("vO", ""),
                                 "issue": item.get("issue", ""),
                                 "page_range": item.get("page_range", ""),
                                 "doc_id": item.get("doc_id", ""),
                                 "kb_id": item.get("kb_id", ""),
                                 "chunk_type": item.get("chunk_type", ""),
                                 "content_with_weight": item.get("content_with_weight", ""),
                                 "pdf_url": item.get("pdf_url", ""),
                                 "level": item.get("level", ""),
                                 "subject": item.get("subject", ""),
                                 "impact_factor": item.get("impact_factor", ""),
                                 "reference": item.get("reference", ""),
                                 "docnm_kwd": item.get("docnm_kwd", ""),
                                 "translated_abstract": item.get("translated_abstract", ""),
                                 "language": item.get("language", "")
                             } for item in raw_results
                         }.values())
    }

    return {
        "results": results,
        "count": len(results["chunks"])
    }


@app.post("/api/search_literature", summary="搜索文献（支持分页）")
def search_literature_api(request: SearchLiteratureRequest):
    """
    文献检索搜索接口，支持分页

    Args:
        request: 包含搜索查询和各种过滤条件的请求对象

    Returns:
        {
            "results": 分页结果,
            "total": 总数,
            "page": 当前页,
            "page_size": 每页数量,
            ...
        }
    """
    raw_results = es_service.search_literature(
        query=request.query,
        year_range=request.year_range,
        journals=request.journals,
        authors=request.authors,
        language=request.language,
        levels=request.levels,
        page_size=int(request.page_size),
        page=int(request.page)
    )

    return raw_results


@app.post("/api/summarize_abstracts", summary="总结多篇研究摘要")
def summarize_abstracts_api(request: AbstractsSummarizeRequest):
    """
    使用大模型对多篇研究摘要进行总结

    Args:
        user_id: 用户ID
        abstracts: 研究摘要列表，格式为[{"author": "作者", "year": "年份", "abstract": "摘要内容"}]

    Returns:
        {
            "status": "success/fail",
            "abstracts_count": 摘要数量,
            "summary": "总结内容"
        }
    """
    user_id = request.user_id
    abstracts = request.abstracts

    logger.info(f"API - 用户 {user_id} 发起摘要总结请求，摘要数量: {len(abstracts)}")

    try:
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

        # 调用总结函数
        summary = summarize_abstracts(abstracts)

        if summary:
            logger.info(f"API - 用户 {user_id} 摘要总结成功，总结长度: {len(summary)}")
            return {
                "status": "success",
                "abstracts_count": len(abstracts),
                "summary": summary
            }
        else:
            logger.warning(f"API - 用户 {user_id} 摘要总结结果为空")
            return {
                "status": "success",
                "abstracts_count": len(abstracts),
                "summary": "",
                "message": "总结生成为空，请检查输入的摘要内容"
            }

    except ValueError as e:
        logger.error(f"API - 用户 {user_id} 输入验证失败: {e}")
        return {
            "status": "fail",
            "abstracts_count": len(abstracts) if isinstance(abstracts, list) else 0,
            "summary": "",
            "message": f"输入验证失败: {str(e)}"
        }

    except Exception as e:
        logger.critical(f"API - 用户 {user_id} 摘要总结异常: {e}", exc_info=True)
        return {
            "status": "fail",
            "abstracts_count": len(abstracts) if isinstance(abstracts, list) else 0,
            "summary": "",
            "message": f"摘要总结失败: {str(e)}"
        }


@app.post("/api/generate_followup", summary="生成追问问题")
def generate_followup(request: FollowupRequest):
    """
    基于对话内容生成追问问题

    Args:
        dialogue_dict: {
            "user": "合并后的所有用户发言内容",
            "assistant": "合并后的所有助手回复内容"
        }

    Returns:
        包含追问的字符串列表
    """
    user_id = request.user_id
    dialogue = request.dialogue

    # 输入验证
    if not all(key in dialogue for key in ["user", "assistant"]):
        logger.warning(f"API - 用户 {user_id} 提交无效对话格式")
        raise HTTPException(
            status_code=400,
            detail="对话必须包含user和assistant字段"
        )

    logger.info(f"API - 用户 {user_id} 发起追问生成")

    try:
        questions = generate_followup_questions(dialogue)

        logger.info(f"API - 用户 {user_id} 生成追问成功: {questions}")
        return {
            "status": "success",
            "questions": questions,
        }

    except ValueError as e:
        logger.error(f"API - 用户 {user_id} 输入验证失败: {e}")
        return {
            "status": "fail",
            "questions": [],
        }

    except Exception as e:
        logger.critical(f"API - 用户 {user_id} 追问生成异常: {e}", exc_info=True)
        return {
            "status": "fail",
            "questions": [],
        }


@app.post("/api/generate_title", summary="生成对话标题")
def generate_title_api(request: DialogueRequest):
    """
    基于用户和助手的对话内容生成标题

    Args:
        request: 包含用户ID和对话内容的请求

    Returns:
        生成的对话标题
    """
    user_id = request.user_id
    dialogue = request.dialogue

    logger.info(f"API - 用户 {user_id} 发起标题生成请求")

    try:
        # 验证对话格式
        if not isinstance(dialogue, dict) or "user" not in dialogue or "assistant" not in dialogue:
            logger.error(f"API - 用户 {user_id} 请求中的对话格式不正确")
            raise HTTPException(
                status_code=400,
                detail="对话格式不正确，必须包含'user'和'assistant'字段"
            )

        # 调用标题生成函数
        title = generate_dialogue_title(dialogue)
        logger.info(f"API - 用户 {user_id} 标题生成成功: {title}")

        return {
            "status": "success",
            "user_id": user_id,
            "title": title
        }
    except Exception as e:
        logger.error(f"API - 用户 {user_id} 标题生成失败: {e}")
        raise HTTPException(status_code=500, detail=f"标题生成失败: {str(e)}")


# ==================== 数据对话API ====================

@app.post("/api/generate_research_data", summary="查找研究数据")
def api_generate_research_data(request: ReviewRequest, background_tasks: BackgroundTasks):
    """
    异步生成研究数据Excel文件

    Args:
        request: 包含查询文本和用户ID的请求
        background_tasks: FastAPI的后台任务管理器

    Returns:
        任务ID和状态信息
    """
    logger.info(f"API - 用户 {request.user_id} 发起找数据，输入为：{request.query}")

    # 生成任务ID
    task_id = f"data_{uuid.uuid4()}"

    # 添加生成任务到后台
    background_tasks.add_task(
        async_generate_research_data,
        query=request.query,
        user_id=request.user_id,
        task_id=task_id
    )

    return {
        "task_id": task_id,
        "user_id": request.user_id,
        "status": "已提交",
        "submit_time": datetime.now().isoformat(),
        "task_type": "research_data"
    }


@app.post("/api/chat_with_data", summary="基于数据回答用户查询")
def chat_with_data_api(request: ChatWithDataRequest):
    """
    使用用户提供的数据回答查询问题

    Args:
        request: 包含用户ID、查询内容、数据JSON和元数据的请求

    Returns:
        基于数据分析的回答结果
    """
    user_id = request.user_id
    query = request.query
    data_json = request.data_json
    format_type = request.format_type
    metadata = request.metadata

    logger.info(f"API - 用户 {user_id} 发起数据对话，查询内容: {query}")

    try:
        # 将JSON字符串转换回DataFrame
        if format_type == "records":
            df = pd.DataFrame(json.loads(data_json))
        elif format_type == "split":
            df = pd.read_json(data_json, orient='split')
        elif format_type == "csv":
            df = pd.read_csv(io.StringIO(data_json))
        else:
            df = pd.read_json(data_json, orient=format_type)

        logger.info(f"API - 成功将JSON数据转换为DataFrame，形状: {df.shape}")

        # 初始化数据分析器
        analyzer = ChatWithData(df, metadata)

        # 执行分析
        result = analyzer.run(query)

        logger.info(f"API - 用户 {user_id} 数据对话成功完成")
        return {"status": "success", "result": result}

    except Exception as e:
        logger.error(f"API - 用户 {user_id} 数据对话失败: {e}")
        raise HTTPException(status_code=500, detail=f"数据对话处理失败: {str(e)}")


@app.post("/api/summarize_data", summary="总结AI相关指标")
def summarize_data_api(request: MetricsAnalysisRequest):
    """
    分析AI相关指标数据，生成关于各指标所代表变量的总结报告

    Args:
        user_id: 用户ID
        metadata: 指标元数据

    Returns:
        {
            "status": "success/fail",
            "indicators_count": 指标数量,
            "summary": "总结内容"
        }
    """
    user_id = request.user_id
    metadata = request.metadata

    logger.info(f"API - 用户 {user_id} 发起AI指标总结请求，指标数量: {len(metadata)}")

    try:
        summary = summarize_data(metadata)

        if summary:
            logger.info(f"API - 用户 {user_id} AI指标总结成功")
            return {
                "status": "success",
                "indicators_count": len(metadata),
                "summary": summary
            }
        else:
            logger.warning(f"API - 用户 {user_id} AI指标总结结果为空")
            return {
                "status": "success",
                "indicators_count": len(metadata),
                "summary": "",
                "message": "总结生成为空"
            }

    except ValueError as e:
        logger.error(f"API - 用户 {user_id} 输入验证失败: {e}")
        return {
            "status": "fail",
            "indicators_count": len(metadata) if isinstance(metadata, list) else 0,
            "summary": "",
            "message": f"输入验证失败: {str(e)}"
        }

    except Exception as e:
        logger.critical(f"API - 用户 {user_id} AI指标总结异常: {e}", exc_info=True)
        return {
            "status": "fail",
            "indicators_count": len(metadata) if isinstance(metadata, list) else 0,
            "summary": "",
            "message": f"AI指标总结失败: {str(e)}"
        }


# ==================== 通用工具API ====================

@app.post("/api/get_task_state", summary="查询任务状态")
def get_task_state_api(request: StatusRequest):
    """
    查询任务的状态和结果

    Args:
        request: 包含用户ID、任务ID和任务类型的请求

    Returns:
        任务状态和结果信息
    """
    user_id = request.user_id
    task_id = request.task_id

    # 从任务ID推断任务类型
    if task_id.startswith("data_"):
        task_type = "research_data"
    elif task_id.startswith("review_"):
        task_type = "review"
    elif task_id.startswith("hypothesis_"):
        task_type = "hypothesis"
    elif task_id.startswith("questionnaire_data_"):
        task_type = "ai_data"
    elif task_id.startswith("download_"):
        task_type = "batch_download"
    else:
        task_type = "unknown"

    # 如果请求中明确指定了任务类型，则使用请求中的类型
    if hasattr(request, 'task_type') and request.task_type:
        task_type = request.task_type

    logger.info(f"API - 用户 {user_id} 查询任务状态, 任务ID: {task_id}, 任务类型: {task_type}")

    # 调用服务函数获取任务状态
    result = get_task_state_service(user_id, task_id, task_type)

    # 处理可能的错误
    if result["status_code"] == -1:
        logger.warning(f"API - 用户 {user_id} 请求的任务未找到")
        raise HTTPException(status_code=404, detail="任务未找到")

    if result["status_code"] == -2:
        logger.error(f"API - 查询出错: {result.get('error', '')}")
        raise HTTPException(status_code=500, detail=f"查询出错: {result.get('error', '')}")

    logger.info(f"API - 返回任务状态: {result['status']}")

    return result


@app.post("/chat", summary="LLM对话接口")
async def chat_endpoint(request: ChatRequest):
    """
    统一的LLM对话接口，支持多个提供商

    Args:
        request: 包含provider、prompt、model等参数的请求

    Returns:
        LLM响应（支持流式和非流式）
    """
    if request.stream:
        # 流式响应
        generator = llm_handler.call_llm(
            provider=request.provider,
            prompt=request.prompt,
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stream=True,
            system_prompt=request.system_prompt,
            history=request.history
        )

        return StreamingResponse(
            generator,
            media_type="text/plain"
        )
    else:
        # 标准响应
        result = llm_handler.call_llm(
            provider=request.provider,
            prompt=request.prompt,
            model=request.model,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            stream=False,
            system_prompt=request.system_prompt,
            history=request.history
        )

        return {"result": result}


# ==================== 启动应用 ====================

def main():
    """启动FastAPI服务器"""
    uvicorn.run(app, host="0.0.0.0", port=8001)


if __name__ == "__main__":
    main()