"""Utilities for rasterizing PDFs into image frames for OCR."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Iterable, List

from pdf2image import convert_from_bytes


@dataclass
class RasterizedPDF:
    """Container that tracks temporary image paths and manages cleanup."""

    image_paths: List[Path]
    _tmpdir: TemporaryDirectory

    def cleanup(self) -> None:
        self._tmpdir.cleanup()


def pdf_bytes_to_images(pdf_bytes: bytes, dpi: int = 220) -> RasterizedPDF:
    """Convert PDF bytes into on-disk PNGs ready for OCR consumption."""

    tmpdir = TemporaryDirectory(prefix="invoice_pages_")
    tmp_path = Path(tmpdir.name)
    pil_images = convert_from_bytes(pdf_bytes, dpi=dpi)
    image_paths: List[Path] = []

    for idx, image in enumerate(pil_images, start=1):
        out_path = tmp_path / f"page_{idx:03d}.png"
        image.save(out_path, format="PNG")
        image_paths.append(out_path)

    return RasterizedPDF(image_paths=image_paths, _tmpdir=tmpdir)


def iter_image_bytes(paths: Iterable[Path]) -> Iterable[bytes]:
    """Yield raw bytes for each PNG path in order."""

    for path in paths:
        yield path.read_bytes()
