FROM python:3.11

WORKDIR /app

# HuggingFace / OpenCLIP キャッシュ固定
ENV HF_HOME=/root/.cache/huggingface
ENV OPENCLIP_CACHE_DIR=/root/.cache/open_clip

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 先に inference を置く（モデルロード用）
COPY inference.py .

# ★ ここで一度モデルをロードして重みをDLさせる
RUN python - <<EOF
from inference import CLIP
CLIP()
print("CLIP weights cached")
EOF

# アプリ本体
COPY main.py .

CMD ["python", "-u", "main.py"]