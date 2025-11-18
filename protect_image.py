#!/usr/bin/env python3
"""
protect_image.py

Combined adversarial + robust DCT watermark pipeline with ETA printing.

Usage:
    python protect_image.py --help

Notes:
- This script intentionally targets CLIP + Face surrogate encoders and embeds
  a spread-spectrum DCT watermark. It prints ETA estimates for the adversarial
  phases so you can monitor progress on environments like Colab.
- Tested conceptually with CLIP, facenet-pytorch and LPIPS. Adjust params for your GPU.
"""

import argparse
import os
import math
import random
import time
from io import BytesIO
from collections import deque

import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as T

# third-party imports that must be installed:
# pip install ftfy regex tqdm
# pip install lpips facenet-pytorch
# pip install git+https://github.com/openai/CLIP.git
import clip
from facenet_pytorch import InceptionResnetV1
import lpips
from scipy.fftpack import dct, idct

# -----------------------
# Configuration / device
# -----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------
# Utilities
# -----------------------
def pil_to_tensor(img):
    return T.ToTensor()(img).unsqueeze(0).to(device)

def tensor_to_pil(t):
    t = t.detach().cpu().squeeze(0).clamp(0,1)
    return T.ToPILImage()(t)

def save_jpeg(img_pil, path, q=95):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    img_pil.save(path, format="JPEG", quality=q)

def ensure_rgb(img):
    if img.mode != "RGB":
        return img.convert("RGB")
    return img

# -----------------------
# ETA helper
# -----------------------
class ETA:
    """
    Small ETA printer that uses a rolling average of step durations.
    """
    def __init__(self, smoothing=20):
        self.smoothing = smoothing
        self.window = deque(maxlen=smoothing)
        self.start = None
        self.last_time = None
        self.steps_done = 0

    def start_timer(self):
        self.start = time.time()
        self.last_time = self.start
        self.steps_done = 0
        self.window.clear()

    def step(self):
        now = time.time()
        if self.last_time is not None:
            dt = now - self.last_time
            self.window.append(dt)
        self.last_time = now
        self.steps_done += 1

    def elapsed(self):
        if self.start is None:
            return 0.0
        return time.time() - self.start

    def get_eta(self, total_steps):
        # if no smoothing samples, just estimate by uniform average
        if not self.window:
            if self.steps_done == 0:
                return None
            avg = (time.time() - self.start) / float(self.steps_done)
        else:
            avg = sum(self.window) / len(self.window)
        remaining = max(total_steps - self.steps_done, 0)
        eta_s = remaining * avg
        return eta_s

    def format_eta(self, total_steps):
        eta = self.get_eta(total_steps)
        if eta is None:
            return "ETA: unknown"
        if eta < 1.0:
            return "ETA: <1s"
        m, s = divmod(int(eta), 60)
        h, m = divmod(m, 60)
        if h > 0:
            return f"ETA: {h}h {m}m {s}s"
        if m > 0:
            return f"ETA: {m}m {s}s"
        return f"ETA: {s}s"

# -----------------------
# DCT block helpers
# -----------------------
def block_process(channel, block_size=8, func=None):
    h, w = channel.shape
    out = np.zeros_like(channel, dtype=np.float32)
    for by in range(0, h, block_size):
        for bx in range(0, w, block_size):
            block = channel[by:by+block_size, bx:bx+block_size]
            bh, bw = block.shape
            if bh != block_size or bw != block_size:
                pad = np.zeros((block_size, block_size), dtype=block.dtype)
                pad[:bh, :bw] = block
                block = pad
            out_block = func(block)
            out[by:by+block_size, bx:bx+block_size] = out_block[:bh, :bw]
    return out

def dct2(block):
    return dct(dct(block.T, norm='ortho').T, norm='ortho')

def idct2(block):
    return idct(idct(block.T, norm='ortho').T, norm='ortho')

# -----------------------
# Spread-spectrum DCT watermark embedding
# -----------------------
def embed_dct_watermark(img_pil, key="secret", alpha=5.0, block_size=8, mid_idx_range=(2,4), redundancy=6, jpeg_q=90):
    """
    Embed a spread-spectrum noise pattern in the Y (luma) channel DCT mid-frequencies.
    """
    img = ensure_rgb(img_pil)
    W, H = img.size
    ycbcr = img.convert('YCbCr')
    y, cb, cr = ycbcr.split()
    y_np = np.array(y).astype(np.float32)
    seed = sum([ord(c) for c in key]) & 0xffffffff
    rng = np.random.RandomState(seed)

    def process_block(block):
        B = dct2(block)
        seq = rng.randn(*(block.shape))
        # add noise to a ring of mid frequencies
        for i in range(block.shape[0]):
            for j in range(block.shape[1]):
                radius = i + j
                if mid_idx_range[0] <= radius <= mid_idx_range[1]:
                    B[i,j] += alpha * seq[i,j]
        out = idct2(B)
        return out

    y_emb = block_process(y_np, block_size=block_size, func=process_block)
    y_emb = np.clip(y_emb, 0, 255).astype(np.uint8)
    out_img = Image.merge("YCbCr", (Image.fromarray(y_emb), cb, cr)).convert("RGB")
    # final JPEG recompression to ensure robustness through quantization
    buf = BytesIO()
    out_img.save(buf, format='JPEG', quality=jpeg_q)
    buf.seek(0)
    out_final = Image.open(buf).convert("RGB")
    return out_final

def detect_dct_watermark_score(img_pil, key="secret", block_size=8, mid_idx_range=(2,4)):
    """
    Compute a correlation-based score for the DCT spread-spectrum watermark.
    Higher is more likely watermark present. Threshold must be calibrated.
    """
    img = ensure_rgb(img_pil)
    ycbcr = img.convert('YCbCr')
    y = np.array(ycbcr.split()[0]).astype(np.float32)
    seed = sum([ord(c) for c in key]) & 0xffffffff
    rng = np.random.RandomState(seed)

    corr_sum = 0.0
    count = 0
    def proc(block):
        nonlocal corr_sum, count
        B = dct2(block)
        vec = []
        mask_coords = []
        for i in range(block.shape[0]):
            for j in range(block.shape[1]):
                radius = i+j
                if mid_idx_range[0] <= radius <= mid_idx_range[1]:
                    vec.append(B[i,j])
                    mask_coords.append((i,j))
        vec = np.array(vec)
        if vec.size == 0:
            return block
        seq = rng.randn(block.shape[0], block.shape[1])
        seq_vec = np.array([seq[i,j] for (i,j) in mask_coords])
        corr = np.dot(vec.flatten(), seq_vec)
        corr_sum += corr
        count += 1
        return block

    _ = block_process(y, block_size=block_size, func=proc)
    if count == 0:
        return 0.0
    return corr_sum / float(count + 1e-9)

# -----------------------
# Helper: high-pass mask
# -----------------------
def high_pass_mask(arr, keep_ratio=0.6):
    """
    Create a high-pass frequency mask (via FFT) to retain high frequencies in the perturbation.
    arr: HxWxC numpy array (values roughly 0..1)
    keep_ratio: fraction of high frequencies to keep (0..1)
    """
    H, W, C = arr.shape
    out = np.zeros_like(arr)
    for ch in range(C):
        F = np.fft.rfft2(arr[:,:,ch])
        fy = np.fft.fftfreq(H)[:,None]
        fx = np.fft.rfftfreq(W)[None,:]
        radius = np.sqrt(fx*fx + fy*fy)
        rmin, rmax = radius.min(), radius.max()
        radius_norm = (radius - rmin) / (rmax - rmin + 1e-12)
        mask = (radius_norm > (1.0 - keep_ratio)).astype(float)
        F_masked = F * mask
        out[:,:,ch] = np.fft.irfft2(F_masked, s=(H,W))
    return out

# -----------------------
# Adversarial perturbation (CLIP + Face surrogate)
# -----------------------
def make_adversarial_clip(orig_pil, eps=8.0, steps=300, num_aug=4, lr=0.02, high_freq_keep=0.6, hf_mix=0.9, eta_obj=None, debug_prefix="adv"):
    """
    Create an adversarial perturbation that reduces similarity in CLIP space and FaceNet space.
    - eps: L_inf in image pixel space (0..255)
    - steps: number of optimization steps
    - num_aug: number of augmentations per step (augmentation builds robustness)
    - lr: optimizer learning rate
    - high_freq_keep: fraction of high freq to keep in perturbation (0..1) when applying HF mask
    - hf_mix: mixing factor for HF component
    - eta_obj: optional ETA object for printing ETA
    - debug_prefix: label printed in logs
    """
    orig_pil = ensure_rgb(orig_pil)
    # load models
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    lpips_model = lpips.LPIPS(net='alex').to(device)

    orig_t = pil_to_tensor(orig_pil)
    _,_,H,W = orig_t.shape

    with torch.no_grad():
        prep = preprocess(T.Resize((224,224))(orig_pil)).unsqueeze(0).to(device)
        ref_emb = clip_model.encode_image(prep)

        fn_in = T.Resize((160,160))(orig_pil)
        emb_face = facenet(T.ToTensor()(fn_in).unsqueeze(0).to(device))

    delta = torch.zeros_like(orig_t, requires_grad=True, device=device)
    eps_f = eps / 255.0
    optimizer = torch.optim.Adam([delta], lr=lr)

    def aug_batch(tensor, n):
        outs = []
        for _ in range(n):
            pil = tensor_to_pil(tensor)
            # spatial augment
            scale = random.uniform(0.85, 1.0)
            w,h = pil.size
            th, tw = int(h*scale), int(w*scale)
            small = pil.resize((tw,th), Image.BICUBIC)
            canvas = Image.new("RGB", (w,h), (int(127+random.uniform(-6,6)),)*3)
            x = random.randint(0, max(0, w-tw))
            y = random.randint(0, max(0, h-th))
            canvas.paste(small, (x,y))
            # JPEG quality randomization
            buf = BytesIO()
            q = random.randint(40,95)
            canvas.save(buf, format='JPEG', quality=q)
            buf.seek(0)
            p2 = Image.open(buf).convert('RGB')
            t2 = pil_to_tensor(p2)
            # small gaussian noise sometimes
            if random.random() < 0.35:
                t2 = (t2 + torch.randn_like(t2) * (random.uniform(1/255, 4/255))).clamp(0,1)
            outs.append(t2)
        return torch.cat(outs, dim=0)

    # ETA setup
    eta = eta_obj or ETA()
    eta.start_timer()

    # main optimization loop
    for step in range(steps):
        optimizer.zero_grad()
        pert = (orig_t + delta).clamp(0,1)
        # build a small augmentation batch for robustness
        aug = aug_batch(pert, num_aug).to(device)
        pil_list = [T.ToPILImage()(aug[i].cpu()) for i in range(aug.shape[0])]
        batch = torch.cat([preprocess(T.Resize((224,224))(p)).unsqueeze(0) for p in pil_list], dim=0).to(device)
        emb = clip_model.encode_image(batch)
        sim = F.cosine_similarity(emb, ref_emb.repeat(emb.shape[0],1), dim=-1).mean()

        fn_batch = torch.stack([T.ToTensor()(T.Resize((160,160))(p)) for p in pil_list]).to(device)
        embf = facenet(fn_batch)
        simf = F.cosine_similarity(embf, emb_face.repeat(embf.shape[0],1), dim=-1).mean()

        sim_mean = (sim + simf) / 2.0

        pert_vis = (orig_t + delta).clamp(0,1)
        lpips_val = lpips_model((orig_t*2-1), (pert_vis*2-1))
        tv = torch.mean(torch.abs(delta[:,:,1:,:]-delta[:,:,:-1,:])) + torch.mean(torch.abs(delta[:,:,:,1:]-delta[:,:,:,:-1]))

        # We want to MINIMIZE similarity -> use sim_mean as positive term to push down
        loss = sim_mean + 0.75 * lpips_val.mean() + 1e-3 * tv
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            # clip by eps
            delta.data = torch.clamp(delta.data, -eps_f, eps_f)
            # optionally push energy to high-freq only
            if high_freq_keep < 1.0:
                d_np = delta.detach().cpu().numpy()[0].transpose(1,2,0)
                hf = high_pass_mask(d_np, keep_ratio=high_freq_keep)
                hf_t = torch.from_numpy(hf.transpose(2,0,1)).unsqueeze(0).to(device)
                delta.data = hf_t * hf_mix + (1.0 - hf_mix) * delta.data
            delta.data = torch.clamp(orig_t + delta.data, 0,1) - orig_t

        # ETA update
        eta.step()
        if step % max(1, steps//10) == 0 or step == steps-1:
            with torch.no_grad():
                pert_vis = (orig_t + delta).clamp(0,1)
                prep = preprocess(T.Resize((224,224))(tensor_to_pil(pert_vis))).unsqueeze(0).to(device)
                emb_p = clip_model.encode_image(prep)
                s = float(F.cosine_similarity(emb_p, ref_emb, dim=-1).item())
            print(f"[{debug_prefix}] step {step}/{steps} sim={s:.4f} | elapsed={int(eta.elapsed())}s {eta.format_eta(steps)}")

    out_pil = tensor_to_pil((orig_t + delta).clamp(0,1))
    return out_pil

# -----------------------
# Main pipeline and CLI
# -----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="input image path")
    parser.add_argument("--out", required=True, help="output image path")
    parser.add_argument("--key", default="secretkey", help="watermark key")
    parser.add_argument("--do_adv", action="store_true", help="run adversarial perturbation first")
    parser.add_argument("--do_watermark", action="store_true", help="embed robust DCT watermark")
    parser.add_argument("--eps", type=float, default=10.0, help="L_inf for adv (0..255)")
    parser.add_argument("--steps", type=int, default=400, help="adv steps")
    parser.add_argument("--num_aug", type=int, default=4, help="augmentations per step")
    parser.add_argument("--lr", type=float, default=0.02, help="optimizer lr")
    parser.add_argument("--alpha", type=float, default=6.0, help="watermark strength alpha")
    parser.add_argument("--jpeg_q", type=int, default=90, help="final JPEG quality")
    parser.add_argument("--fine_tune_steps", type=int, default=120, help="optional fine-tune adv steps after watermarking")
    parser.add_argument("--high_freq_keep", type=float, default=0.6, help="keep ratio for high-frequency mask (0..1)")
    parser.add_argument("--hf_mix", type=float, default=0.9, help="mixing factor for HF masking")
    parser.add_argument("--redundancy", type=int, default=6, help="watermark redundancy (unused currently)")
    args = parser.parse_args()

    img = Image.open(args.input).convert("RGB")
    cur = img

    # create a single ETA object to show each phase ETA
    eta_main = ETA(smoothing=25)

    if args.do_adv:
        if device == "cpu":
            print("Warning: no CUDA device; adversarial step will be slow.")
        print("Running adversarial perturbation ...")
        # print estimated wall-clock based on small micro-batch test (cheap heuristic)
        try:
            # warmup short run to estimate speed: single step with tiny num_aug
            warmup_steps = min(2, max(1, args.num_aug))
            t0 = time.time()
            _ = make_adversarial_clip(cur, eps=min(args.eps, 8.0), steps=2, num_aug=1, lr=args.lr,
                                      high_freq_keep=args.high_freq_keep, hf_mix=args.hf_mix,
                                      eta_obj=eta_main, debug_prefix="warmup")
            warmup_dt = time.time() - t0
            est_total = (args.steps / 2.0) * warmup_dt if warmup_dt > 0 else None
            if est_total:
                m, s = divmod(int(est_total), 60)
                h, m = divmod(m, 60)
                print(f"[estimate] main adv approx: {h}h {m}m {s}s (rough)")
        except Exception:
            # ignore warmup failure and run full
            pass

        cur = make_adversarial_clip(cur, eps=args.eps, steps=args.steps, num_aug=args.num_aug, lr=args.lr,
                                    high_freq_keep=args.high_freq_keep, hf_mix=args.hf_mix,
                                    eta_obj=eta_main, debug_prefix="adv")
        print("Adversarial step done.")

    if args.do_watermark:
        print("Embedding DCT watermark")
        cur = embed_dct_watermark(cur, key=args.key, alpha=args.alpha, mid_idx_range=(2,4), redundancy=args.redundancy, jpeg_q=args.jpeg_q)
        print("Watermark Embedded.")

    if args.fine_tune_steps > 0:
        print("Running fine-tune adversarial pass to restore encoder-hostile effect ...")
        eta_ft = ETA(smoothing=25)
        cur = make_adversarial_clip(cur, eps=args.eps, steps=args.fine_tune_steps, num_aug=max(1, min(6, args.num_aug//2)),
                                    lr=args.lr*1.2, high_freq_keep=args.high_freq_keep, hf_mix=args.hf_mix,
                                    eta_obj=eta_ft, debug_prefix="fine-tune")
        print("Fine-tune done.")

    save_jpeg(cur, args.out, q=args.jpeg_q)
    print("Saved final protected image to:", args.out)

if __name__ == "__main__":
    main()
