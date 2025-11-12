1. StegaStamp repo: https://github.com/avinashpaliwal/StegaStamp (or tancik's implementation). Clone it.
2. Use their encoder to embed a short token (e.g., "no_ai=1;owner=you") into the defended image.
3. After embedding, re-run a short defense fine-tune (200–400 steps) on the final embedded image to restore encoder-breaking effect because the stega embedding may slightly change pixels.
4. Save the final image. Keep the stega decoder in your server for later detection, or use it to create signed manifests (C2PA compatibility).
