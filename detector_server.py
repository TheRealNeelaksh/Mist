#!/usr/bin/env python3
"""
detector_server.py

Simple, fast detector for the DCT watermark produced by protect_image.py.
Use it on the server side (or your own checks) to automatically reject images
that contain your secret watermark pattern.

Example:
    python detector_server.py --input protected.jpg --key mysecret --threshold 1e6
"""

import argparse
from PIL import Image
from protect_image import detect_dct_watermark_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Image path to analyze")
    parser.add_argument("--key", default="secretkey", help="Watermark key")
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e6,
        help="Reject if score > threshold (calibrate using real data)",
    )
    args = parser.parse_args()

    img = Image.open(args.input).convert("RGB")

    score = detect_dct_watermark_score(img, key=args.key)
    print(f"DCT watermark score: {score:.4f}")

    if score > args.threshold:
        print("RESULT: WATERMARK DETECTED → REJECT")
        exit(2)
    else:
        print("RESULT: CLEAN → ALLOW")
        exit(0)


if __name__ == "__main__":
    main()
