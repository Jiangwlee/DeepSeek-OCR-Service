ARG BASE_IMAGE=paddlepaddle/paddle:3.2.2
FROM ${BASE_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=Asia/Shanghai

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxrender1 libxext6 libgomp1 tzdata \
    && ln -snf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime \
    && echo "Asia/Shanghai" >/etc/timezone \
    && rm -rf /var/lib/apt/lists/*

# Python 依赖
ARG REQUIREMENTS_FILE=services/paddle-ocr/requirements.cpu.txt
COPY ${REQUIREMENTS_FILE} ./requirements.txt
RUN pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple && \
    pip install --upgrade pip && pip install -r requirements.txt && \
    rm -rf /root/.cache/pip

COPY services/paddle-ocr/app ./app

EXPOSE 9000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9000"]