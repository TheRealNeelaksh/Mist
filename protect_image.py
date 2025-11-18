#!/usr/bin/env python3
"""
protect_image.py

Combined adversarial + robust DCT watermark pipeline (production-ready-ish).
Run with: python protect_image.py --help
"""
import argparse
import os
import math
import random
from io import BytesIO
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import clip
from facenet_pytorch import InceptionResnetV1
import lpips
from scipy.fftpack import dct, idct

# -----------------------
# Utilities
# -----------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

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
    buf = BytesIO()
    out_img.save(buf, format='JPEG', quality=jpeg_q)
    buf.seek(0)
    out_final = Image.open(buf).convert("RGB")
    return out_final

# -----------------------
# DCT watermark detector
# -----------------------
def detect_dct_watermark_score(img_pil, key="secret", block_size=8, mid_idx_range=(2,4)):
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
        coords = []
        for i in range(block.shape[0]):
            for j in range(block.shape[1]):
                radius = i+j
                if mid_idx_range[0] <= radius <= mid_idx_range[1]:
                    vec.append(B[i,j])
                    coords.append((i,j))
        vec = np.array(vec)
        if vec.size == 0:
            return block
        seq = rng.randn(*block.shape)
        seq_vec = np.array([seq[i,j] for (i,j) in coords])
        corr = np.dot(vec.flatten(), seq_vec)
        corr_sum += corr
        count += 1
        return block

    _ = block_process(y, block_size=block_size, func=proc)
    if count == 0:
        return 0.0

    return corr_sum / (count + 1e-9)

# -----------------------
# Adversarial noise (CLIP + face)
# -----------------------
def make_adversarial_clip(orig_pil, eps=8.0, steps=300, num_aug=4, lr=0.02,
                          high_freq_keep=0.6, hf_mix=0.9):

    orig_pil = ensure_rgb(orig_pil)
    clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    lpips_model = lpips.LPIPS(net='alex').to(device)

    orig_t = pil_to_tensor(orig_pil)
    with torch.no_grad():
        prep = preprocess(T.Resize((224,224))(orig_pil)).unsqueeze(0).to(device)
        ref_emb = clip_model.encode_image(prep)
        fn_in = T.Resize((160,160))(orig_pil)
        emb_face = facenet(T.ToTensor()(fn_in).unsqueeze(0).to(device))

    delta = torch.zeros_like(orig_t, requires_grad=True)
    eps_norm = eps/255.0
    optimizer = torch.optim.Adam([delta], lr=lr)

    def aug_batch(tensor, n):
        out = []
        for _ in range(n):
            pil = tensor_to_pil(tensor)
            scale = random.uniform(0.8, 1.0)
            w,h = pil.size
            tw,th = int(w*scale), int(h*scale)
            sm = pil.resize((tw,th), Image.BICUBIC)
            canvas = Image.new("RGB", (w,h), (128,128,128))
            x = random.randint(0, max(0,w-tw))
            y = random.randint(0, max(0,h-th))
            canvas.paste(sm,(x,y))
            buf = BytesIO()
            canvas.save(buf, format="JPEG", quality=random.randint(40,95))
            buf.seek(0)
            p2 = Image.open(buf).convert("RGB")
            t2 = T.ToTensor()(p2).unsqueeze(0)
            if random.random() < 0.25:
                t2 = (t2 + torch.randn_like(t2)*0.01).clamp(0,1)
            out.append(t2)
        return torch.cat(out).to(device)

    for step in range(steps):
        optimizer.zero_grad()
        pert = (orig_t + delta).clamp(0,1)

        aug = aug_batch(pert, num_aug)
        pil_list = [T.ToPILImage()(aug[i].cpu()) for i in range(aug.size(0))]
        batch = torch.cat([
            preprocess(T.Resize((224,224))(p)).unsqueeze(0)
            for p in pil_list
        ]).to(device)

        emb = clip_model.encode_image(batch)
        sim_clip = F.cosine_similarity(emb, ref_emb.repeat(emb.size(0),1), dim=-1).mean()

        fn_batch = torch.stack([T.ToTensor()(T.Resize((160,160))(p)) for p in pil_list]).to(device)
        emb_f = facenet(fn_batch)
        sim_face = F.cosine_similarity(emb_f, emb_face.repeat(emb_f.size(0),1), dim=-1).mean()

        sim = (sim_clip + sim_face)/2
        lp = lpips_model((orig_t*2-1),(pert*2-1)).mean()

        tv = torch.mean(torch.abs(delta[:,:,1:,:]-delta[:,:,:-1,:])) + \
             torch.mean(torch.abs(delta[:,:,:,1:]-delta[:,:,:,:-1]))

        loss = sim + 0.75*lp + 1e-3*tv
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            delta.data = torch.clamp(delta.data, -eps_norm, eps_norm)

        if step % max(1, steps//10) == 0:
            print(f"[adv] step {step}/{steps} sim={sim.item():.4f}")

    return tensor_to_pil((orig_t+delta).clamp(0,1))

# -----------------------
# Main
# -----------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--key", default="secretkey")
    p.add_argument("--do_adv", action="store_true")
    p.add_argument("--do_watermark", action="store_true")
    p.add_argument("--eps", type=float, default=10.0)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--num_aug", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--alpha", type=float, default=6.0)
    p.add_argument("--jpeg_q", type=int, default=90)
    p.add_argument("--fine_tune_steps", type=int, default=120)
    args = p.parse_args()

    img = Image.open(args.input).convert("RGB")
    cur = img

    if args.do_adv:
        cur = make_adversarial_clip(
            cur, eps=args.eps, steps=args.steps,
            num_aug=args.num_aug, lr=args.lr
        )

    if args.do_watermark:
        print("Embedding DCT watermark")
        cur = embed_dct_watermark(
            cur, key=args.key, alpha=args.alpha, jpeg_q=args.jpeg_q
        )
        print("Watermark Embedded.")

    if args.fine_tune_steps > 0:
        cur = make_adversarial_clip(
            cur, eps=args.eps, steps=args.fine_tune_steps,
            num_aug=3, lr=args.lr*1.5
        )

    save_jpeg(cur, args.out, q=args.jpeg_q)
    print("Saved:", args.out)

if __name__ == "__main__":
    main()
