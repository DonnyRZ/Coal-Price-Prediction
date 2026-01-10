# 使用官方轻量级 Python 镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 设置时区为上海
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 安装基础系统依赖 (精简版)
# git 是为了某些 pip 包，curl 用于健康检查
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 1. 复制精简版依赖文件
COPY requirements_prod.txt .

# 2. 安装依赖 (指定文件名)
RUN pip install --no-cache-dir -r requirements_prod.txt

# 3. 复制项目代码
COPY . .

# 暴露端口
EXPOSE 8501

# 启动命令
ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]