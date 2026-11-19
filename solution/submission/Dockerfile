# syntax=docker/dockerfile:1
# Harness 提交镜像：平台用 `docker save` 导出 Linux/amd64 未压缩 tar 上传。
# 构建：docker build --platform linux/amd64 -t contest-harness:1.0 -f Dockerfile .
FROM python:3.12-slim-bookworm

LABEL org.opencontainers.image.title="rsi-contest-harness" \
      org.opencontainers.image.description="工业模型自进化 Harness（A/B/C，CPU，NumPy）" \
      org.opencontainers.image.version="1.0"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    CONTEST_WORKSPACE=/workspace \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN python3 -m pip install --no-cache-dir --disable-pip-version-check -r requirements.txt

# 平台固定入口 + 选手模块 + 模型代码 + 官方 SDK（vendored，版本 1.0.2）
COPY app/harness.py ./harness.py
COPY app/contest_submission.py ./contest_submission.py
COPY app/contest_core.py ./contest_core.py
COPY app/experiment_log.py ./experiment_log.py
COPY app/contest_harness ./contest_harness

RUN mkdir -p /workspace \
    && python3 -c "import numpy, contest_harness, contest_submission; print('image import check ok')"

ENTRYPOINT ["python3", "/app/harness.py"]
CMD ["describe"]
