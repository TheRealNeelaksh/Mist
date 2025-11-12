# utils.py - helper IO, metrics, transforms
import os
from PIL import Image
import numpy as np
import math
from skimage.metrics import structural_similarity as ssim
import torch
import torchvision.transforms as T
import lpips

# Image IO
def load_image(path):
    img = Image.open(path).convert("RGB")
    return img

def save_image(img_tensor, path):
    # expects tensor or PIL
    if isinstance(img_tensor, torch.Tensor):
        img = T.ToPILImage()(img_tensor.clamp(0,1).cpu().squeeze(0))
    else:
        img = img_tensor
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)

# Basic metrics: PSNR, SSIM
def psnr_np(a, b):
    # a,b float arrays 0..1
    mse = float(((a - b) ** 2).mean())
    return 10 * math.log10(1.0 / mse) if mse > 0 else float("inf")

def ssim_np(a, b):
    # expects HxWx3 arrays 0..1
    # skimage expects 0..255 or float 0..1; multichannel True
    return ssim(a, b, data_range=1.0, multichannel=True)

# LPIPS wrapper
_lpips_model = None
def lpips_distance(tensor_a, tensor_b, device):
    # expects tensors in 0..1 shape [1,3,H,W]
    global _lpips_model
    if _lpips_model is None:
        _lpips_model = lpips.LPIPS(net='alex').to(device).eval()
    a = tensor_a * 2.0 - 1.0
    b = tensor_b * 2.0 - 1.0
    with torch.no_grad():
        return float(_lpips_model(a, b).mean().cpu().item())
