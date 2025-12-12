import io
import open_clip
import torch
from PIL import Image
import numpy as np

class CLIP:
    def __init__(self, model_name="ViT-B-32", pretrained="laion2b_s34b_b79k"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # OpenCLIP のモデル & preprocess を取得
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained
        )

        self.model.to(self.device)
        self.model.eval()

    def encode_batch(self, images):
        tensors = []
        for arr in images:
            pil = Image.fromarray(arr.astype(np.uint8))
            tensors.append(self.preprocess(pil))
        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            vecs = self.model.encode_image(batch)
        vecs = vecs / vecs.norm(dim=-1, keepdim=True)
        return vecs.cpu().numpy()

clip = CLIP()