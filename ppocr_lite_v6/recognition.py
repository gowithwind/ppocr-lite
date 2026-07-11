"""Text recognition using PP-OCR CTC models.

Pre-processing: PIL resize + numpy normalise.
Post-processing: CTC greedy decode from the ONNX model's embedded character list.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Pre-processing
# ---------------------------------------------------------------------------

class RecPreProcess:
    """Resize to fixed height (48) and normalise.

    PP-OCR recognition models use rec_img_shape = [3, 48, W].
    Normalisation: ``x/127.5 - 1`` (same for v5 and v6 recognition).
    """

    HEIGHT = 48

    def __init__(self, max_width: int = 320) -> None:
        """
        Parameters
        ----------
        max_width:
            Maximum output width. v5: 320, v6 PP-OCR: no cap (pass large value
            like 9999 or use ``max_width=None`` for unlimited).
        """
        self.MAX_WIDTH = max_width if max_width is not None else 99999

    def resize_norm(self, img: np.ndarray, max_wh_ratio: float) -> np.ndarray:
        """Resize *img* to (3, HEIGHT, img_w) and normalise.

        *img_w* is determined by ``max_wh_ratio`` across the current batch
        (matching PaddleOCR's batching logic) but capped at MAX_WIDTH.

        Normalisation is ``x/127.5 - 1``.
        """
        h, w = img.shape[:2]
        # Use ceil to avoid truncation: img_w must be >= resized_w
        img_w    = min(int(math.ceil(self.HEIGHT * max_wh_ratio)), self.MAX_WIDTH)
        resized_w = min(img_w, int(math.ceil(self.HEIGHT * w / h)))

        pil = Image.fromarray(img).resize((resized_w, self.HEIGHT), Image.BILINEAR)
        # Normalise: x/127.5 − 1  (same for v5 and v6 recognition)
        arr = np.asarray(pil, dtype=np.float32) * (1.0 / 127.5) - 1.0
        arr = arr.transpose(2, 0, 1)  # → CHW (3, H, W)

        # When resized width equals img_w, model doesn't see padding — skip it.
        # This is critical for v6: padding changes CNN receptive field output.
        if resized_w == img_w:
            return np.ascontiguousarray(arr)

        out = np.zeros((3, self.HEIGHT, img_w), dtype=np.float32)
        out[:, :, :resized_w] = arr
        return out


# ---------------------------------------------------------------------------
# CTC greedy decode
# ---------------------------------------------------------------------------

class CTCDecoder:
    """Greedy CTC decoder with blank-token removal and duplicate-collapse.

    Two modes:
      v5: ``["blank", *chars, " "]``  — blank at 0, chars at 1..N, space at N+1.
      v6: ``["", *chars]`` with last char treated as space  — blank at 0,
          chars at 1..N, N is space.  Matches PaddleOCR convention.
    """

    def __init__(self, characters: List[str], model_version: str = "v5") -> None:
        self.model_version = model_version
        if model_version == "v6":
            # PaddleOCR convention: index 0 = empty (blank), last = space
            self.chars = ["", *characters]
            self.space_idx = len(self.chars) - 1
        else:
            self.chars = ["blank", *characters, " "]
            self.space_idx = -1  # space is a real character at end

    @classmethod
    def from_model_metadata(cls, meta: dict, model_version: str = "v5") -> "CTCDecoder":
        raw = meta.get("character", "")
        chars = raw.splitlines()
        return cls(chars, model_version=model_version)

    @classmethod
    def from_file(cls, path: Path, model_version: str = "v5") -> "CTCDecoder":
        chars = path.read_bytes().decode("utf-8").splitlines()
        return cls(chars, model_version=model_version)

    def decode(
        self, preds: np.ndarray
    ) -> List[Tuple[str, float]]:
        """Decode a batch of CTC outputs.

        Parameters
        ----------
        preds:
            Shape (N, T, C) float32.

        Returns
        -------
        List of (text, mean_confidence) pairs.
        """
        indices = preds.argmax(axis=2)   # (N, T)
        probs   = preds.max(axis=2)      # (N, T)
        results = []
        n_chars = len(self.chars)

        for idx_seq, prob_seq in zip(indices, probs):
            out_chars: List[str] = []
            confs: List[float] = []
            prev = 0  # blank index
            for tok, p in zip(idx_seq.tolist(), prob_seq.tolist()):
                if tok == 0:          # blank — resets duplicate suppression
                    prev = 0
                    continue
                if tok == prev:       # consecutive duplicate → collapse
                    continue
                prev = tok
                if tok < n_chars:
                    ch = self.chars[tok]
                    if self.model_version == "v6" and tok == self.space_idx:
                        # Model's last-index = space
                        out_chars.append(" ")
                    elif ch:
                        out_chars.append(ch)
                    # else: empty string (blank-like, skip)
                    confs.append(p)
            text  = "".join(out_chars)
            score = sum(confs) / len(confs) if confs else 0.0
            results.append((text, round(score, 5)))

        return results
