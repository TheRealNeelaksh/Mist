# defend.py
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageFilter
import torchvision.transforms as T
import clip
from facenet_pytorch import InceptionResnetV1
import random
import os
from utils import load_image, save_image, lpips_distance
import kornia.augmentation as K
import math

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Utilities (local)
# -------------------------
def pil_to_tensor(img):
    return T.ToTensor()(img).unsqueeze(0).to(device)  # [1,3,H,W] 0..1

def tensor_to_pil(t):
    t = t.detach().cpu().squeeze(0).clamp(0,1)
    return T.ToPILImage()(t)

def high_pass_mask_fft(delta, keep_ratio=0.6):
    # delta: torch tensor [1,3,H,W] on cpu or device
    # returns same shape limited to high-frequency bands
    delta_np = delta.detach().cpu().numpy()[0]  # [3,H,W]
    c, H, W = delta_np.shape
    out = np.zeros_like(delta_np)
    for ch in range(c):
        f = np.fft.rfft2(delta_np[ch])
        fy = np.fft.fftfreq(H)[:,None]
        fx = np.fft.rfftfreq(W)[None,:]
        radius = np.sqrt(fx*fx + fy*fy)
        rmin, rmax = radius.min(), radius.max()
        radius_norm = (radius - rmin) / (rmax - rmin + 1e-12)
        mask = (radius_norm > (1.0 - keep_ratio)).astype(float)
        f_masked = f * mask
        out[ch] = np.fft.irfft2(f_masked, s=(H,W))
    out = torch.from_numpy(out).unsqueeze(0).to(delta.device)
    return out

# augmentations for robustness
def random_jpeg(tensor, qmin=40, qmax=95):
    # tiny helper - convert to PIL and re-encode
    pil = T.ToPILImage()(tensor.clamp(0,1).cpu().squeeze(0))
    from io import BytesIO
    buf = BytesIO()
    q = random.randint(qmin, qmax)
    pil.save(buf, format='JPEG', quality=q)
    buf.seek(0)
    return pil_to_tensor(Image.open(buf))

def screen_sim(tensor):
    pil = tensor_to_pil(tensor)
    W, H = pil.size
    # perspective warp
    def rand_pt(x,y,scale=0.02):
        return (x + random.uniform(-scale,scale)*W, y + random.uniform(-scale,scale)*H)
    coeffs = T.functional._get_perspective_coeffs(
        [(0,0),(W,0),(W,H),(0,H)],
        [rand_pt(0,0),(W,0),(W,H),(0,H)]
    )
    pil = pil.transform((W,H), Image.PERSPECTIVE, coeffs, Image.BICUBIC)
    pil = pil.filter(ImageFilter.GaussianBlur(radius=random.uniform(0,1.2)))
    if random.random() < 0.5:
        from io import BytesIO
        buf = BytesIO()
        q = random.randint(30,90)
        pil.save(buf, format='JPEG', quality=q)
        buf.seek(0)
        return pil_to_tensor(Image.open(buf))
    return pil_to_tensor(pil)

def augmentation_batch(tensor, num=4):
    out = []
    for _ in range(num):
        t = tensor
        # random scale and paste like phone screenshot variance
        scale = random.uniform(0.7, 1.0)
        _,_,H,W = t.shape
        th, tw = int(H*scale), int(W*scale)
        pil = T.ToPILImage()(t.cpu().squeeze(0))
        pil = pil.resize((tw, th), Image.BICUBIC)
        canvas = Image.new("RGB", (W,H), (int(127+random.uniform(-8,8)),)*3)
        x_off = random.randint(0, W-tw) if W>tw else 0
        y_off = random.randint(0, H-th) if H>th else 0
        canvas.paste(pil, (x_off, y_off))
        t = pil_to_tensor(canvas)
        if random.random() < 0.5:
            t = random_jpeg(t, 30, 95)
        if random.random() < 0.25:
            t = screen_sim(t)
        if random.random() < 0.25:
            t = pil_to_tensor(tensor_to_pil(t).filter(ImageFilter.GaussianBlur(radius=random.uniform(0,1.2))))
        if random.random() < 0.25:
            noise = torch.randn_like(t) * (random.uniform(1/255,4/255))
            t = (t + noise).clamp(0,1)
        out.append(t)
    return torch.cat(out, dim=0)

# -------------------------
# Main defense routine
# -------------------------
def run_defense(args):
    # load models
    print("Loading surrogate models...")
    clip_vit, preprocess_vit = clip.load("ViT-B/32", device=device, jit=False)
    clip_rn, preprocess_rn = clip.load("RN50", device=device, jit=False)
    facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    # load image
    pil = load_image(args.input)
    orig = pil_to_tensor(pil)  # [1,3,H,W]
    _,_,H,W = orig.shape

    # compute original embeddings
    with torch.no_grad():
        prep1 = preprocess_vit(T.Resize((224,224))(pil)).unsqueeze(0).to(device)
        emb_vit_o = clip_vit.encode_image(prep1)
        prep2 = preprocess_rn(T.Resize((224,224))(pil)).unsqueeze(0).to(device)
        emb_rn_o = clip_rn.encode_image(prep2)
        fn_in = T.Resize((160,160))(pil)
        fn_in = T.ToTensor()(fn_in).unsqueeze(0).to(device)
        emb_face_o = facenet(fn_in)

    # delta init
    delta = torch.zeros_like(orig, requires_grad=True, device=device)
    eps = args.eps / 255.0
    optimizer = torch.optim.Adam([delta], lr=args.lr)

    best = None

    for step in range(args.steps):
        optimizer.zero_grad()
        pert = (orig + delta).clamp(0,1)

        # create augmentations
        aug = augmentation_batch(pert, num=args.num_aug).to(device)  # [B,3,H,W]

        # encode each augmentation for both CLIP models and face net
        # CLIP expects preprocess pipeline: do per-aug PIL preprocess
        aug_pils = [T.ToPILImage()(aug[i].cpu()) for i in range(aug.shape[0])]
        clip_inputs_vit = torch.cat([preprocess_vit(T.Resize((224,224))(p)).unsqueeze(0) for p in aug_pils], dim=0).to(device)
        clip_inputs_rn  = torch.cat([preprocess_rn(T.Resize((224,224))(p)).unsqueeze(0) for p in aug_pils], dim=0).to(device)

        emb_vit = clip_vit.encode_image(clip_inputs_vit)
        emb_rn  = clip_rn.encode_image(clip_inputs_rn)

        # face embeddings
        fn_inputs = torch.stack([T.ToTensor()(T.Resize((160,160))(p)) for p in aug_pils]).to(device)
        emb_face = facenet(fn_inputs)

        # similarity (cosine) to original
        sim1 = (F.cosine_similarity(emb_vit, emb_vit_o.repeat(emb_vit.shape[0],1), dim=-1)).mean()
        sim2 = (F.cosine_similarity(emb_rn, emb_rn_o.repeat(emb_rn.shape[0],1), dim=-1)).mean()
        simf = (F.cosine_similarity(emb_face, emb_face_o.repeat(emb_face.shape[0],1), dim=-1)).mean()

        # objective: minimize similarity
        sim_mean = (sim1 + sim2 + simf) / 3.0

        # perceptual distance penalty (LPIPS)
        pert_vis = (orig + delta).clamp(0,1)
        p_loss = lpips_distance(orig, pert_vis, device)

        # TV regularizer to smooth delta
        tv = torch.mean(torch.abs(delta[:,:,1:,:]-delta[:,:,:-1,:])) + torch.mean(torch.abs(delta[:,:,:,1:]-delta[:,:,:,:-1]))

        loss = sim_mean + args.lambda_percept * p_loss + args.lambda_tv * tv

        # backprop (note: LPIPS returns float; we want differentiable version - for simplicity we use it as a constant penalty here).
        # If you want full differentiable LPIPS loss in training, import lpips and use model(a,b) directly in graph.
        # For current pipeline, we keep optimization guided mainly by sim_mean and tv and use LPIPS as monitor.
        loss.backward()
        optimizer.step()

        # project to L_inf
        with torch.no_grad():
            delta.data = torch.clamp(delta.data, -eps, eps)
            # optional high-frequency projection
            if args.high_freq_keep < 1.0:
                hf = high_pass_mask_fft(delta.data, keep_ratio=args.high_freq_keep)
                # mixing ratio to keep delta stable
                alpha = args.hf_mix
                delta.data = alpha*hf + (1-alpha)*delta.data
            delta.data = torch.clamp(orig + delta.data, 0,1) - orig

        if step % args.log_interval == 0 or step == args.steps-1:
            # compute clean similarities for reporting
            with torch.no_grad():
                cp = preprocess_vit(T.Resize((224,224))(tensor_to_pil(pert_vis))).unsqueeze(0).to(device)
                emb_vit_p = clip_vit.encode_image(cp)
                cp2 = preprocess_rn(T.Resize((224,224))(tensor_to_pil(pert_vis))).unsqueeze(0).to(device)
                emb_rn_p = clip_rn.encode_image(cp2)
                fn_p = T.ToTensor()(T.Resize((160,160))(tensor_to_pil(pert_vis))).unsqueeze(0).to(device)
                emb_face_p = facenet(fn_p)

                s1 = float(F.cosine_similarity(emb_vit_p, emb_vit_o, dim=-1).item())
                s2 = float(F.cosine_similarity(emb_rn_p, emb_rn_o, dim=-1).item())
                sf = float(F.cosine_similarity(emb_face_p, emb_face_o, dim=-1).item())

                print(f"step {step}/{args.steps} sim_clean mean={(s1+s2+sf)/3:.4f} clip_vit={s1:.4f} clip_rn={s2:.4f} face={sf:.4f} lpips={p_loss:.4f}")

                mean_sim = (s1+s2+sf)/3.0
                if best is None or mean_sim < best[0]:
                    best = (mean_sim, pert_vis.detach().cpu().clone())

    # save best
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, args.out_name)
    save_image(best[1], out_path)
    print("Saved defended image to:", out_path)

def tensor_to_pil(t):
    t = t.detach().cpu().squeeze(0).clamp(0,1)
    return T.ToPILImage()(t)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="input image path (JPG/PNG)")
    parser.add_argument("--out_dir", default="outputs", help="output dir")
    parser.add_argument("--out_name", default="defended.png")
    parser.add_argument("--eps", type=float, default=8.0, help="L_inf epsilon in 0..255 scale")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--num_aug", type=int, default=6)
    parser.add_argument("--lambda_percept", type=float, default=0.5, help="LPIPS weight used in printed logs")
    parser.add_argument("--lambda_tv", type=float, default=1e-3)
    parser.add_argument("--high_freq_keep", type=float, default=0.6, help="fraction high frequencies to keep (0..1)")
    parser.add_argument("--hf_mix", type=float, default=0.9, help="mix ratio for HF projection")
    parser.add_argument("--log_interval", type=int, default=50)
    args = parser.parse_args()
    run_defense(args)
