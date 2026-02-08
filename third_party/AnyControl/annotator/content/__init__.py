import cv2
import numpy as np
import torch
from annotator.util import annotator_ckpts_path
from PIL import Image
from transformers import AutoProcessor, CLIPModel


class ContentDetector:
    def __init__(self, model_name="openai/clip-vit-large-patch14"):

        self.model = CLIPModel.from_pretrained(model_name, cache_dir=annotator_ckpts_path).cuda().eval()
        self.processor = AutoProcessor.from_pretrained(model_name, cache_dir=annotator_ckpts_path)

    def __call__(self, img):
        with torch.no_grad():
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            inputs = self.processor(images=[img], return_tensors="pt").to('cuda')
            image_features = self.model.get_image_features(**inputs)
            content_emb = image_features[0].detach().cpu().numpy()
        return content_emb
