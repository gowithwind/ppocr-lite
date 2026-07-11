# ppocr-lite

Lightweight PP-OCR runtime — ONNX only, no OpenCV, no heavy frameworks.

Forked from [ppocr_lite](https://pypi.org/project/ppocr-lite/) with added **PP-OCRv6** support.

## Install

```bash
pip3 install git+https://github.com/gowithwind/ppocr-lite.git
```

## Usage

```python
from ppocr_lite import PPOCRLite

# PP-OCRv5（默认）
ocr = PPOCRLite()

# PP-OCRv6 tiny（最快）
ocr = PPOCRLite(model_version="v6")

# PP-OCRv6 small
ocr = PPOCRLite(model_version="v6", v6_size="small")

# PP-OCRv6 medium（最准）
ocr = PPOCRLite(model_version="v6", v6_size="medium")

# OCR 识别
results = ocr.run("screenshot.png")
for r in results:
    print(f"{r.score:.2f}  {r.text}")
```

## Model sizes

| Version | Size | Speed | Quality |
|---------|------|-------|---------|
| v5      | -    | ~2.8s | baseline |
| v6 tiny | 1.7MB + 4.4MB | **~1.4s** | good |
| v6 small | 4.7MB + 13MB | ~4.1s | better |
| v6 medium | 7.5MB + 22MB | ~13.8s | best |

Models are auto-downloaded on first run from CDN.

## Notes

- v6 detection uses `limit_side_len=640` (vs v5's 960)
- v6 recognition has no width cap (vs v5's 320px max)
- Direction classifier not used with v6 (`cls_model=False`)
