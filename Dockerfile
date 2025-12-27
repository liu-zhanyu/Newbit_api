FROM python:3.12-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 复制 requirements.txt
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir httpx==0.24.1

# 复制应用代码 - 按照项目结构
COPY main.py model.py ./
COPY .env .env.example ./

# 复制所有模块文件夹
COPY common/ ./common/
COPY chat_with_paper/ ./chat_with_paper/
COPY chat_with_data/ ./chat_with_data/

# 暴露服务端口
EXPOSE 8001

# 设置环境变量
ENV PYTHONUNBUFFERED=1

# 启动应用
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]