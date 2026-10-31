# CV_2 — Cost-Efficient Segmentation (CSL7360 A2, Track B)

Tiny U-Net salient-object segmenter under a **50 ms single-core CPU** budget on Apple M2.

## Deliverables

| Path | Description |
| --- | --- |
| `predict.py` | Required CLI: resize → segment → time 20 runs |
| `train.py` / `eval.py` | Training and full metrics + \(T_\mathrm{CPU}\) |
| `ces/` | Model, metrics, data, inference helpers |
| `checkpoints/model.pt` | Learned float16 parameters |
| `captured/` | `original/`, `images_256/`, `masks_256/` |
| `REPORT.md` | Method, table, overlays pointer, failures |
| `results/metrics.json` | Computed scores including CES |
| `figures/` | Captured overlays + failure panels |

## Quick start

```bash
source .venv/bin/activate
python predict.py --image captured/images_256/bottle.png --out mask.png
python eval.py --out results/metrics.json
```

## Headline numbers (this MacBook Air M2)

- \(T_\mathrm{CPU}\) ≈ **15.9 ms**
- \(S_\mathrm{disk}\) = **576,923** bytes
- DUTS-TE Q ≈ **0.585** → **CES ≈ 0.540**
- Captured (7 real iPhone photos) Q ≈ **0.605**

See `REPORT.md` for the full write-up, metric table, overlays, and failure cases.
