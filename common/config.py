"""
配置加载器
从.env文件加载所有配置变量
"""
import os
import json
from dotenv import load_dotenv

# 加载.env文件
load_dotenv()

# ==================== MongoDB配置 ====================
MONGO_HOST = os.getenv("MONGO_HOST", "localhost")
MONGO_PORT = int(os.getenv("MONGO_PORT", "27017"))
MONGO_USER = os.getenv("MONGO_USER", "admin")
MONGO_PASSWORD = os.getenv("MONGO_PASSWORD", "")
MONGO_DB = os.getenv("MONGO_DB", "research_db")
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "tasks")

# ==================== Elasticsearch配置 ====================
ES_HOST = os.getenv("ES_HOST", "http://localhost:9200")
ES_USER = os.getenv("ES_USER", "elastic")
ES_PASSWORD = os.getenv("ES_PASSWORD", "")
DEFAULT_INDEX = os.getenv("DEFAULT_INDEX", "ragflow_index")

# ==================== 知识库ID ====================
KB_ID_PAPER = os.getenv("KB_ID_PAPER", "")
KB_ID_CHUNK = os.getenv("KB_ID_CHUNK", "")
KB_ID_SUMMARY = os.getenv("KB_ID_SUMMARY", "")
DATA_INDICATORS_KB_ID = os.getenv("DATA_INDICATORS_KB_ID", "")

# ==================== 搜索参数 ====================
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "30"))
DEFAULT_VECTOR_WEIGHT = float(os.getenv("DEFAULT_VECTOR_WEIGHT", "0.7"))
DEFAULT_TEXT_WEIGHT = float(os.getenv("DEFAULT_TEXT_WEIGHT", "0.3"))

# DEFAULT_CHUNK_TYPE需要从字符串转为列表
chunk_type_str = os.getenv("DEFAULT_CHUNK_TYPE", '["raw"]')
try:
    DEFAULT_CHUNK_TYPE = json.loads(chunk_type_str)
except:
    DEFAULT_CHUNK_TYPE = ["raw"]

# ==================== BGE API配置 ====================
BGE_API_URL = os.getenv("BGE_API_URL", "https://api.siliconflow.cn/v1/embeddings")

# BGE_API_KEYS需要从字符串转为列表
bge_keys_str = os.getenv("BGE_API_KEYS", '[]')
try:
    BGE_API_KEYS = json.loads(bge_keys_str)
except:
    BGE_API_KEYS = []

# ==================== LLM API配置 ====================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY", "")
ARK_API_KEY = os.getenv("ARK_API_KEY", "")
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY", "")
QWEN_API_KEY = os.getenv("QWEN_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

# ==================== OSS配置 ====================
OSS_ENDPOINT = os.getenv("OSS_ENDPOINT", "")
OSS_BUCKET_NAME = os.getenv("OSS_BUCKET_NAME", "")
OSS_ACCESS_KEY_ID = os.getenv("OSS_ACCESS_KEY_ID", "")
OSS_ACCESS_KEY_SECRET = os.getenv("OSS_ACCESS_KEY_SECRET", "")