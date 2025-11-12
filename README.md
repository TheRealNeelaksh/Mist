# Image Defender — adversarial + perceptual image protection

Make images hard for common AI encoders to use while keeping them visually the same for humans.

**What this does**

  * Generates a small, high-frequency adversarial perturbation that *breaks model embeddings* (CLIP, FaceNet) while preserving human-perceived quality.
  * Includes a test harness to measure PSNR / SSIM / LPIPS and embedding similarity before/after defense.
  * Optional: instructions for adding an invisible steganographic token (StegaStamp) for provenance.

**Important ethics & legal note**

  * Use this tool only for protecting your own images or images you have the right to modify.
  * Avoid processing images of people who haven't consented if they are identifiable or sensitive.
  * These defenses increase the cost for misuse but do not guarantee perfect protection against determined attackers.

-----

## Setup

1.  Clone this repo locally.
2.  Create virtualenv and install dependencies (GPU recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

If you have a CUDA GPU, install a CUDA-enabled PyTorch wheel per your system.

**Files:**

  * `defend.py` — main defense script
  * `test_harness.py` — compute PSNR/SSIM/LPIPS & embedding similarities
  * `utils.py` — helper functions
  * `optional/stegastamp_integration.md` — how to add stega token

-----

## Quick usage

### Defend an image

```bash
python defend.py --input path/to/original.jpg --out_dir outputs --out_name defended.png --eps 8.0 --steps 600 --num_aug 6
```

**Key flags:**

  * `--eps`: max pixel change in 0..255 units (6–10 recommended).
  * `--steps`: optimization iterations (400–1200 typical).
  * `--num_aug`: augmentations per step for robustness.
  * `--high_freq_keep`: fraction of high-frequency content to keep (0.6 recommended).

Output: `outputs/defended.png`.

### Test the defended image

```bash
python test_harness.py --orig path/to/original.jpg --defended outputs/defended.png
```

This reports PSNR/SSIM/LPIPS and embedding cosine similarities for CLIP/FaceNet surrogates.

-----

## Tuning advice (practical)

  * Start with `eps=6` or `eps=8`. Inspect visually and run `test_harness.py`.
  * If embeddings are still high, increase `steps` or `eps` gradually to 10–12 (watch image quality).
  * Increase `num_aug` for better robustness to screenshots and resizing.
  * Use `high_freq_keep` around 0.5–0.8 to concentrate perturbation on texture.

-----

## Deployment tips

  * Always host **only the defended version** publicly. Keep originals offline.
  * If you control the upload server, add a server-side detector that decodes an embedded token (StegaStamp/Digimarc/C2PA) and rejects images with `no_ai=1`.
  * Periodically re-generate defenses if you find a platform circumvents them.

-----

## Limitations & countermeasures

  * This raises the bar but is not bulletproof. Skilled adversaries can denoise, crop heavily, re-train encoders, or use alternate pipelines.
  * If a platform aggressively normalizes (heavy denoising / auto-enhance), the perturbation may be weakened. Re-test against target platforms.

-----

## Troubleshooting

  * **OOM on GPU:** reduce batch sizes or model load. Use smaller `num_aug` or reduce `steps`.
  * **Slow:** run fewer `steps` for prototyping; resume longer runs on GPU.
  * **Want better perceptual quality:** enable LPIPS-based loss fully inside optimization (modify `defend.py` to use `lpips.LPIPS(net='alex')` in the computation graph).