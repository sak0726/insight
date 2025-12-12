import runpod
import base64
import io
import numpy as np
from PIL import Image
from inference import clip

def handler(job):
    """
    RunPod Serverlessから呼び出される関数
    job = {
        "input": {
            "images": ["base64_string_1", "base64_string_2", ...]
        }
    }
    """
    job_input = job["input"]
    
    # 入力チェック
    if "images" not in job_input:
        return {"error": "No images provided in input"}

    images_b64 = job_input["images"]
    images = []

    # Base64デコード -> PIL -> numpy
    for b64_str in images_b64:
        try:
            # ヘッダー(data:image/jpeg;base64,)がついている場合の除去処理（念のため）
            if "," in b64_str:
                b64_str = b64_str.split(",")[1]
            
            img_data = base64.b64decode(b64_str)
            img = Image.open(io.BytesIO(img_data)).convert("RGB")
            images.append(np.array(img))
        except Exception as e:
            return {"error": f"Failed to decode image: {str(e)}"}

    if not images:
        return {"vectors": []}

    # inference.py の encode_batch をコール
    vecs = clip.encode_batch(images)

    # 結果を返す (リスト形式)
    return {"vectors": vecs.tolist()}

# Serverlessのエントリーポイント
runpod.serverless.start({"handler": handler})