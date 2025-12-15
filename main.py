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
    job_input = job.get("input", {})

    # 1. 入力チェック（images は list）
    if "images" not in job_input:
        return {"error": "Input must contain 'images' list."}

    b64_list = job_input["images"]
    if not isinstance(b64_list, list):
        return {"error": "'images' must be a list."}

    images_np = []
    image_ids = []

    # 2. 画像データのデコード (Base64 -> PIL -> Numpy)
    for i, item in enumerate(b64_list):
        try:
            # 変数名は変えない
            b64_str = item["image"]
            img_id = item["id"]

            # "data:image/png;base64,..." 対策
            if "," in b64_str:
                b64_str = b64_str.split(",")[1]

            image_data = base64.b64decode(b64_str)
            pil_image = Image.open(io.BytesIO(image_data)).convert("RGB")

            images_np.append(np.array(pil_image))
            image_ids.append(img_id)

        except Exception as e:
            return {
                "error": f"Failed to decode image at index {i}: {str(e)}"
            }

    if not images_np:
        return {"vectors": []}

    # 3. 推論実行（GPU）
    try:
        vectors = clip.encode_batch(images_np)
    except Exception as e:
        return {"error": f"Inference failed: {str(e)}"}

    # 4. ID付きで結果を返却（順序保証）
    results = []
    for img_id, vec in zip(image_ids, vectors):
        results.append({
            "id": img_id,
            "vector": vec.tolist()
        })

    return {"vectors": results}

# Serverlessのエントリーポイント
runpod.serverless.start({"handler": handler})