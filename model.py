"""
Pydantic数据模型
定义所有API请求和响应的数据结构
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


# ==================== 通用模型 ====================

class ClassifyRequest(BaseModel):
    """查询分类请求"""
    user_id: str = Field(..., description="用户ID")
    query: str = Field(..., description="查询内容")


class StatusRequest(BaseModel):
    """任务状态查询请求"""
    user_id: str = Field(..., description="用户ID")
    task_id: str = Field(..., description="任务ID")
    task_type: Optional[str] = Field(default=None, description="任务类型")


# ==================== 文献对话模型 ====================

class SearchRequest(BaseModel):
    """文档搜索请求"""
    query: str = Field(..., description="搜索查询")
    top_k: int = Field(default=10, description="返回结果数量")
    docnm_kwds: Optional[List[str]] = Field(default=None, description="文档名称关键词列表")


class SearchDocumentsRequest(BaseModel):
    """高级文档搜索请求"""
    query: str = Field(..., description="搜索查询")
    top_k: int = Field(default=30, description="返回结果数量")
    year_range: Optional[List[int]] = Field(default=None, description="年份范围 [start, end]")
    kb_id: Optional[str] = Field(default=None, description="知识库ID")
    journals: Optional[List[str]] = Field(default=None, description="期刊列表")
    authors: Optional[List[str]] = Field(default=None, description="作者列表")
    language: Optional[str] = Field(default=None, description="文献语言")


class SearchLiteratureRequest(BaseModel):
    """文献检索请求（支持分页）"""
    query: str = Field(..., description="搜索查询")
    year_range: Optional[List[int]] = Field(default=None, description="年份范围 [start, end]")
    journals: Optional[List[str]] = Field(default=None, description="期刊列表")
    authors: Optional[List[str]] = Field(default=None, description="作者列表")
    language: Optional[str] = Field(default=None, description="文献语言")
    levels: Optional[List[str]] = Field(default=None, description="层级列表")
    page: int = Field(default=1, description="页码")
    page_size: int = Field(default=10, description="每页数量")


class QueryRewriteRequest(BaseModel):
    """查询改写请求"""
    user_id: str = Field(..., description="用户ID")
    history: List[Dict[str, str]] = Field(..., description="历史对话记录")
    query: str = Field(..., description="当前查询")


class AbstractsSummarizeRequest(BaseModel):
    """摘要总结请求"""
    user_id: str = Field(..., description="用户ID")
    abstracts: List[Dict[str, Any]] = Field(..., description="摘要列表")


class FollowupRequest(BaseModel):
    """追问生成请求"""
    user_id: str = Field(..., description="用户ID")
    dialogue: Dict[str, str] = Field(..., description="对话内容 {user: ..., assistant: ...}")


class DialogueRequest(BaseModel):
    """对话标题生成请求"""
    user_id: str = Field(..., description="用户ID")
    dialogue: Dict[str, str] = Field(..., description="对话内容 {user: ..., assistant: ...}")


# ==================== 数据对话模型 ====================

class ReviewRequest(BaseModel):
    """研究数据生成请求"""
    user_id: str = Field(..., description="用户ID")
    query: str = Field(..., description="研究查询")


class ChatWithDataRequest(BaseModel):
    """数据对话请求"""
    user_id: str = Field(..., description="用户ID")
    query: str = Field(..., description="查询内容")
    data_json: str = Field(..., description="数据JSON字符串")
    format_type: str = Field(default="records", description="JSON格式类型")
    metadata: List[Dict[str, Any]] = Field(..., description="数据元信息")


class MetricsAnalysisRequest(BaseModel):
    """指标分析请求"""
    user_id: str = Field(..., description="用户ID")
    metadata: List[Dict[str, Any]] = Field(..., description="指标元数据")


# ==================== LLM对话模型 ====================

class ChatRequest(BaseModel):
    """LLM对话请求"""
    provider: str = Field(..., description="LLM提供商")
    prompt: str = Field(..., description="提示词")
    model: str = Field(default="gpt-4o-mini", description="模型名称")
    max_tokens: int = Field(default=1000, description="最大token数")
    temperature: float = Field(default=0.7, description="温度参数")
    stream: bool = Field(default=False, description="是否流式输出")
    system_prompt: Optional[str] = Field(default=None, description="系统提示词")
    history: Optional[List[Dict[str, str]]] = Field(default=None, description="对话历史")


# ==================== 响应模型 ====================

class BaseResponse(BaseModel):
    """基础响应模型"""
    status: str = Field(..., description="状态：success/fail")
    message: Optional[str] = Field(default=None, description="消息")


class SearchResponse(BaseModel):
    """搜索响应"""
    results: Dict[str, Any] = Field(..., description="搜索结果")
    count: int = Field(..., description="结果数量")


class TaskResponse(BaseModel):
    """任务响应"""
    task_id: str = Field(..., description="任务ID")
    user_id: str = Field(..., description="用户ID")
    status: str = Field(..., description="任务状态")
    submit_time: str = Field(..., description="提交时间")
    task_type: str = Field(..., description="任务类型")


class TaskStateResponse(BaseModel):
    """任务状态响应"""
    status_code: int = Field(..., description="状态码")
    status: str = Field(..., description="任务状态")
    task_id: str = Field(..., description="任务ID")
    result: Optional[Dict[str, Any]] = Field(default=None, description="任务结果")
    error: Optional[str] = Field(default=None, description="错误信息")