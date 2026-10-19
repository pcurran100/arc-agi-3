"""The frame as a PNG for multimodal models, alongside (never instead of) exact text.

D-042 reverses the "no PNG for a vision model" convention in one respect: the
image is *added* to the exact facts, because a model's prior about what a
picture shows (a door, a key, a maze) is the intuition we want from it, and the
text alone hides it. Every fact the agent acts on still comes from the exact
fact layer; the image only informs the model's guesses.

Colours are the official ARC-AGI-3 palette (arc_agi.rendering.COLOR_MAP), so
the picture looks like the game a human plays. Stdlib only (zlib), so it runs in
the offline Kaggle notebook.
"""

from __future__ import annotations

import base64
import struct
import zlib

import numpy as np

PALETTE = np.array([
    (0xFF, 0xFF, 0xFF), (0xCC, 0xCC, 0xCC), (0x99, 0x99, 0x99), (0x66, 0x66, 0x66),
    (0x33, 0x33, 0x33), (0x00, 0x00, 0x00), (0xE5, 0x3A, 0xA3), (0xFF, 0x7B, 0xCC),
    (0xF9, 0x3C, 0x31), (0x1E, 0x93, 0xFF), (0x88, 0xD8, 0xF1), (0xFF, 0xDC, 0x00),
    (0xFF, 0x85, 0x1B), (0x92, 0x12, 0x31), (0x4F, 0xCC, 0x30), (0xA3, 0x56, 0xD6),
], dtype=np.uint8)

COLOUR_NAMES = ("white", "off-white", "light grey", "grey", "dark grey", "black", "magenta",
                "pink", "red", "blue", "light blue", "yellow", "orange", "maroon", "green",
                "purple")


def legend() -> str:
    return "COLOURS: " + ", ".join(f"{i}={n}" for i, n in enumerate(COLOUR_NAMES))


def png_bytes(grid: np.ndarray, scale: int = 8) -> bytes:
    g = np.clip(np.asarray(grid, dtype=np.int16), 0, 15)
    rgb = PALETTE[g].repeat(scale, axis=0).repeat(scale, axis=1)
    h, w = rgb.shape[:2]
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def image_part(grid: np.ndarray, scale: int = 8) -> dict:
    """An OpenAI-style ``image_url`` content part holding the frame."""
    b64 = base64.b64encode(png_bytes(grid, scale)).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
