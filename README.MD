# Newbit 学术研究助手 API

> 基于 RAG (Retrieval-Augmented Generation) 架构的学术研究智能服务系统，提供文献对话和数据对话两大核心功能，并支持深度解读 (Deep Chat) 增强分析。

---

## 目录

- [功能特性](#功能特性)
- [系统架构](#系统架构)
- [项目结构](#项目结构)
- [工作流程](#工作流程)
- [快速开始](#快速开始)
- [API文档](#api文档)
- [配置说明](#配置说明)
- [部署指南](#部署指南)

---

## 功能特性

### 核心功能

#### 文献对话 (Chat with Paper)
- **智能检索**：混合检索（向量+文本）+ 学术参数提取
- **查询优化**：查询分类、改写、翻译、消歧
- **摘要生成**：自动总结多篇文献，提取核心观点
- **智能追问**：基于对话历史生成相关追问
- **深度解读**：对特定文献进行深入分析和解读

#### 数据对话 (Chat with Data)
- **变量提取**：自动识别核心变量（2-3个）和控制变量（5-8个）
- **指标推荐**：为每个变量推荐3个测量指标
- **数据检索**：智能搜索并批量获取研究数据（year≥2010）
- **统计分析**：线性回归、相关性、描述性统计、分组统计、亚组分析
- **深度分析**：基于数据的深入洞察，使用 ChatPoint API 生成分析报告
- **数据导出**：异步生成Excel文件并上传到OSS

### 技术特性

- RAG 架构：检索增强生成，确保回答准确性
- 多模态检索：向量检索 + 文本检索 + RRF融合
- 深度解读：独特的 Deep Chat 模块，调用 ChatPoint API 提供更深入的分析
- 异步处理：支持长时任务的后台处理
- 多LLM支持：OpenAI、Claude、智谱AI、字节ARK等多个提供商

---

## 系统架构

传统 RAG 流程：
```
用户请求 → 预处理 → 检索 → 生成 → 后处理 → 返回结果
```

增强 Deep Chat 流程：
```
传统 RAG 流程 → Deep Chat (深度解读) → 增强结果
```

两大模块：
1. **chat_with_paper** - 文献对话模块
2. **chat_with_data** - 数据对话模块

---

## 项目结构
```
Newbit_api/
│
├── main.py                      # FastAPI 应用主入口
├── model.py                     # Pydantic 数据模型定义
├── requirements.txt             # Python 依赖包
├── Dockerfile                   # Docker 构建文件
├── .env                         # 环境变量配置（不提交Git）
├── .env.example                 # 环境变量示例
├── README.md                    # 项目文档
│
├── common/                      # 公共组件层
│   ├── __init__.py
│   ├── config.py               # 配置加载器（从 .env 读取）
│   ├── llm_call.py             # LLM 调用封装（多提供商支持）
│   ├── es_connector.py         # Elasticsearch 连接器（混合检索）
│   ├── mongodb_connector.py    # MongoDB 连接器（数据存储）
│   ├── oss_handler.py          # 阿里云 OSS 文件处理
│   └── state_checker.py        # 任务状态检查器
│
├── chat_with_paper/            # 文献对话模块
│   ├── __init__.py
│   │
│   ├── preprocessing/          # 阶段1：预处理
│   │   ├── __init__.py
│   │   ├── query_classifier.py        # 查询分类（0:QA型 / 1:数据型 / 2:混合型）
│   │   ├── query_rewriter.py          # 查询改写（基于对话历史消歧）
│   │   ├── query_optimizer.py         # 查询优化（生成最优检索词）
│   │   ├── query_translator.py        # 查询翻译（中英互译）
│   │   ├── academic_param_extractor.py # 学术参数提取（年份/期刊/作者/语言）
│   │   └── chunk_type_mapper.py       # 文档类型映射
│   │
│   ├── retrieval/              # 阶段2：检索
│   │   ├── __init__.py
│   │   └── academic_searcher.py       # 学术检索（hybrid_search_v4 多阶段混合检索）
│   │
│   ├── deep_chat/              # 阶段3：生成（调用 ChatPoint API）
│   │   ├── __init__.py
│   │   ├── abstract_summarizer.py     # 摘要总结生成（LLM生成总结）
│   │   └── document_searcher.py       # 简单文档搜索（基础检索）
│   │
│   └── postprocessing/         # 阶段4：后处理
│       ├── __init__.py
│       ├── followup_generator.py      # 追问生成（基于对话生成3个追问）
│       └── title_generator.py         # 对话标题生成
│
└── chat_with_data/             # 数据对话模块
    ├── __init__.py
    │
    ├── preprocessing/          # 阶段1：预处理
    │   ├── __init__.py
    │   ├── topic_analyzer.py          # 主题分析（VariableExtractionAgent）
    │   ├── variable_extractor.py      # 变量提取（核心2-3个 + 控制5-8个）
    │   └── indicator_suggester.py     # 指标推荐（每变量推荐3个测量指标）
    │
    ├── retrieval/              # 阶段2：检索
    │   ├── __init__.py
    │   ├── indicator_searcher.py      # 指标搜索（ES混合检索）
    │   └── data_retriever.py          # 数据检索（MongoDB批量获取year≥2010数据）
    │
    ├── deep_chat/              # 阶段3：生成（调用 ChatPoint API）
    │   ├── __init__.py
    │   ├── data_summarizer.py         # 数据总结生成（区分核心指标和辅助指标）
    │   └── data_analyzer.py           # 数据深度分析
    │                                  #   - Tools类：5种统计分析方法
    │                                  #   - ChatWithData类：OpenAI Function Call智能分析
    │
    └── postprocessing/         # 阶段4：后处理
        ├── __init__.py
        └── research_data_exporter.py  # 研究数据导出（异步生成Excel并上传OSS）
```

---

## 工作流程

### 文献对话 (chat_with_paper)

**传统 RAG 流程：**
```
1. 预处理 (preprocessing/)
   - 查询分类 → 查询改写 → 查询翻译 → 查询优化 → 学术参数提取
   
2. 检索 (retrieval/)
   - academic_searcher.py: 使用 hybrid_search_v4 多阶段混合检索
   - 第1阶段：在 KB_ID_PAPER 中文本检索获取 docnm_kwd
   - 第2阶段：多线程获取 raw chunks 和 non-raw chunks
   - 第3阶段：获取相邻 chunks 并合并内容
   - 第4阶段：过滤（去除"作者简介"、数字占比>30%）并排序
   
3. 生成 (deep_chat/)
   - abstract_summarizer.py: 调用 ChatPoint API 生成文献摘要总结
   - 输入：多篇文献的作者、年份、摘要
   - 输出：200-400字的学术总结，包含引用标注
   
4. 后处理 (postprocessing/)
   - followup_generator.py: 生成3个相关追问
   - title_generator.py: 生成对话标题
```

**Deep Chat 深度解读流程：**
```
传统RAG结果 → 深度解读模块 → 增强分析
   - 针对特定文献进行更深入的内容分析
   - 使用 ChatPoint API 进行专业解读
   - 提供更详细的学术洞察
```

---

### 数据对话 (chat_with_data)

**传统 RAG 流程：**
```
1. 预处理 (preprocessing/)
   - variable_extractor.py: 提取核心变量（2-3个）和控制变量（5-8个）
   - indicator_suggester.py: 为每个变量推荐3个测量指标
   
2. 检索 (retrieval/)
   - indicator_searcher.py: 在ES中搜索数据指标
   - data_retriever.py: 
     * 协调整个检索流程
     * 从MongoDB批量获取数据（year≥2010）
     * 返回 DataFrame 和 metadata
   
3. 生成 (deep_chat/)
   - data_summarizer.py: 调用 ChatPoint API 生成数据总结
     * 区分核心指标和控制变量指标
     * 生成150-250字的专业总结
   
   - data_analyzer.py: 智能数据分析
     * Tools类提供5种分析方法：
       1. linear_regression - 线性回归分析（支持稳健标准误）
       2. correlation_analysis - 相关性分析（Pearson/Spearman/Kendall）
       3. descriptive_statistics - 描述性统计
       4. group_statistics - 分组统计
       5. subgroup_analysis - 亚组分析
     
     * ChatWithData类：
       - classify_query_type: 分类查询类型（元数据查询/数据分析）
       - get_analysis_plan: 使用OpenAI Function Call选择分析方法
       - execute_analysis: 执行具体的统计分析
       - run: 统一入口，返回分析结果的Markdown表格
   
4. 后处理 (postprocessing/)
   - research_data_exporter.py: 异步生成研究数据Excel
     * 生成Excel文件（包含数据和元数据两个sheet）
     * 上传到阿里云OSS
     * 更新MongoDB中的任务状态
```

**Deep Chat 深度分析流程：**
```
传统RAG结果 → 深度分析模块 → 统计分析报告
   - 使用 ChatPoint API 进行数据解读
   - 自动选择合适的统计分析方法
   - 生成专业的分析报告和可视化表格
   - 提供数据洞察和研究建议
```

---

## 快速开始

### 1. 环境要求

- Python 3.12+
- Elasticsearch 8.x
- MongoDB 4.x+
- 阿里云 OSS（可选）

### 2. 安装依赖
```bash
# 克隆项目
git clone 
cd Newbit_api

# 安装依赖
pip install -r requirements.txt
```

### 3. 配置环境变量
```bash
# 复制环境变量示例文件
cp .env.example .env

# 编辑 .env 文件，填写实际配置
# 必填项：
# - MongoDB 连接信息
# - Elasticsearch 连接信息
# - OpenAI API Key
# - OSS 配置（如需文件上传功能）
```

### 4. 启动服务
```bash
# 开发模式
python main.py

# 生产模式
uvicorn main:app --host 0.0.0.0 --port 8001 --workers 4
```

### 5. 访问 API 文档

打开浏览器访问：
- Swagger UI: http://localhost:8001/docs
- ReDoc: http://localhost:8001/redoc

---

## API 文档

### 文献对话 API

#### POST /api/classify_query
查询分类（0:QA型 / 1:数据型 / 2:混合型）

#### POST /api/rewrite_query
查询改写和消歧

#### POST /api/search_with_router_v3
高级学术检索（结合学术参数提取）

#### POST /api/search_documents
智能文献搜索（支持多种过滤条件）

#### POST /api/search_literature
文献检索（支持分页）

#### POST /api/summarize_abstracts
总结多篇研究摘要

#### POST /api/generate_followup
生成追问问题

#### POST /api/generate_title
生成对话标题

---

### 数据对话 API

#### POST /api/generate_research_data
异步生成研究数据Excel文件

#### POST /api/chat_with_data
基于数据回答用户查询（智能分析）

#### POST /api/summarize_data
总结数据指标

---

### 通用 API

#### POST /api/get_task_state
查询异步任务状态

#### POST /chat
LLM对话接口（支持多提供商）

---

## 配置说明

### .env 配置项
```bash
# MongoDB配置
MONGO_HOST=43.134.113.96
MONGO_PORT=27017
MONGO_USER=admin
MONGO_PASSWORD=your_password
MONGO_DB=database_database
MONGO_COLLECTION=Data

# Elasticsearch配置
ES_HOST=http://43.134.113.96:1200
ES_USER=elastic
ES_PASSWORD=your_password
DEFAULT_INDEX=ragflow_index

# 知识库ID
KB_ID_PAPER=302b50b61e7911f0822c0242ac120006
KB_ID_CHUNK=684df8ee1e7511f0a9ff0242ac120006
KB_ID_SUMMARY=3dcd9e360c6811f081000242ac120004
DATA_INDICATORS_KB_ID=ab8652a4f9aa11ef9d410242ac120006

# 搜索参数
DEFAULT_TOP_K=30
DEFAULT_VECTOR_WEIGHT=0.7
DEFAULT_TEXT_WEIGHT=0.3

# LLM API配置
OPENAI_API_KEY=sk-...
CLAUDE_API_KEY=sk-ant-...

# OSS配置
OSS_ENDPOINT=https://oss-cn-qingdao.aliyuncs.com
OSS_BUCKET_NAME=hentre-user-upload
OSS_ACCESS_KEY_ID=your_key
OSS_ACCESS_KEY_SECRET=your_secret
```

详细配置项说明请参考 `.env.example` 文件。

---

## 部署指南

### Docker 部署
```bash
# 构建镜像
docker build -t newbit-api:latest .

# 运行容器
docker run -d \
  --name newbit-api \
  -p 8001:8001 \
  --env-file .env \
  newbit-api:latest

# 查看日志
docker logs -f newbit-api
```

### Docker Compose 部署
```bash
# 启动服务
docker-compose up -d

# 停止服务
docker-compose down
```

### 生产环境部署建议

- 使用 Nginx 作为反向代理
- 配置 HTTPS 证书
- 使用 Supervisor 或 Systemd 管理进程
- 配置日志轮转
- 设置资源限制和监控

---

## 技术栈

**后端框架**
- FastAPI 0.104.1
- Uvicorn (ASGI服务器)

**数据存储**
- MongoDB 4.x+ (任务状态、数据存储)
- Elasticsearch 8.x (文档检索)
- 阿里云 OSS (文件存储)

**AI/ML**
- OpenAI GPT-4 (文本生成)
- Claude (备用LLM)
- BGE-M3 (向量嵌入)
- Jieba (中文分词)

**数据处理**
- Pandas (数据分析)
- statsmodels (统计分析)
- NumPy (数值计算)

---

## 开发指南

### 添加新的分析方法

在 `chat_with_data/deep_chat/data_analyzer.py` 的 `Tools` 类中添加新方法：
```python
class Tools:
    def new_analysis_method(self, ...):
        """新的分析方法"""
        # 实现分析逻辑
        pass
```

然后在 `ChatWithData.get_function_descriptions()` 中注册。

### 添加新的检索策略

在 `common/es_connector.py` 中添加新的搜索方法。

---

## 常见问题

**Q: 如何切换不同的LLM提供商？**

A: 在 `.env` 中配置对应的API Key，在调用时指定 `provider` 参数。

**Q: 数据检索为什么只返回year≥2010的数据？**

A: 在 `data_retriever.py` 中设置，可修改此筛选条件。

**Q: 如何调整向量和文本检索的权重？**

A: 修改 `.env` 中的 `DEFAULT_VECTOR_WEIGHT` 和 `DEFAULT_TEXT_WEIGHT`。

---

## 许可证

MIT License

---

## 联系方式

如有问题或建议，请联系开发团队。