# main.py (OpenCLIP版)

import io
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image

from inference import clip   # ← Pod起動時にロード済みの CLIP インスタンスを使う


app = FastAPI()

@app.post("/encode")
async def encode_images(files: list[UploadFile] = File(...)):
    images = []
    for f in files:
        img = Image.open(io.BytesIO(await f.read())).convert("RGB")
        images.append(np.array(img))

    # full + parts をまとめてバッチ処理
    vecs = clip.encode_batch(images)

    return {"vectors": vecs.tolist()}


@app.get("/health")
def health():
    return {"status": "ok"}
