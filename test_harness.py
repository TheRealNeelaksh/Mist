# test_harness.py
import argparse
import torch
from PIL import Image
import torchvision.transforms as T
import numpy as np
from utils import load_image, psnr_np, ssim_np, lpips_distance
import clip
from facenet_pytorch import InceptionResnetV1
import os
import random
import math

device = "cuda" if torch.cuda.is_available() else "cpu"

def img_to_tensor(img):
    return T.ToTensor()(img).unsqueeze(0).to(device)

def compute_clip_face_embeddings(img_pil, clip_vit, clip_rn, facenet):
    prep_vit = clip_vit.encode_image(clip.load("ViT-B/32")[1](T.Resize((224,224))(img_pil)).unsqueeze(0).to(device))
    return prep_vit

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig", required=True)
    parser.add_argument("--defended", required=True)
    parser.add_argument("--out_dir", default="outputs")
    args = parser.parse_args()

    # load models
    clip_vit, preprocess_vit = clip.load("ViT-B/32", device=device, jit=False)
    clip_rn, preprocess_rn = clip.load("RN50", device=device, jit=False)
    facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    orig = load_image(args.orig)
    def_img = load_image(args.defended)

    # basic metrics
    orig_arr = np.array(orig).astype(np.float32)/255.0
    def_arr = np.array(def_img).astype(np.float32)/255.0
    print("PSNR:", psnr_np(orig_arr, def_arr))
    print("SSIM:", ssim_np(orig_arr, def_arr))
    print("LPIPS:", lpips_distance(img_to_tensor(orig), img_to_tensor(def_img), device))

    # CLIP embeddings (clean)
    with torch.no_grad():
        emb_vit_o = clip_vit.encode_image(preprocess_vit(T.Resize((224,224))(orig)).unsqueeze(0).to(device))
        emb_vit_d = clip_vit.encode_image(preprocess_vit(T.Resize((224,224))(def_img)).unsqueeze(0).to(device))
        emb_rn_o = clip_rn.encode_image(preprocess_rn(T.Resize((224,224))(orig)).unsqueeze(0).to(device))
        emb_rn_d = clip_rn.encode_image(preprocess_rn(T.Resize((224,224))(def_img)).unsqueeze(0).to(device))
        fn_o = T.ToTensor()(T.Resize((160,160))(orig)).unsqueeze(0).to(device)
        fn_d = T.ToTensor()(T.Resize((160,160))(def_img)).unsqueeze(0).to(device)
        emb_face_o = facenet(fn_o)
        emb_face_d = facenet(fn_d)

        def cosine(a,b):
            return float((F.cosine_similarity(a,b,dim=-1)).item())

        sim_vit = F.cosine_similarity(emb_vit_o, emb_vit_d, dim=-1).item()
        sim_rn = F.cosine_similarity(emb_rn_o, emb_rn_d, dim=-1).item()
        sim_face = F.cosine_similarity(emb_face_o, emb_face_d, dim=-1).item()

    print("Embedding cosines (orig vs defended):")
    print("CLIP ViT:", sim_vit)
    print("CLIP RN:", sim_rn)
    print("FaceNet (ArcFace style):", sim_face)

    # optional: run transforms to test robustness (jpeg, crop, resize)
    # You can implement and report here - left as exercise (see README).
