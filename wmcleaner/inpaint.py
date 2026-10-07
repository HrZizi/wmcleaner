from __future__ import annotations

from typing import Literal
from pathlib import Path
import hashlib
import os

import cv2
import numpy as np
from PIL import Image

Backend = Literal["auto", "lama", "anime-lama", "opencv"]
Device = Literal["auto", "cuda", "cpu"]


class LamaUnavailable(RuntimeError):
    pass


ANIME_LAMA_URL = os.environ.get(
    "WMCLEAN_ANIME_LAMA_URL",
    "https://github.com/Sanster/models/releases/download/AnimeMangaInpainting/anime-manga-big-lama.pt",
)
ANIME_LAMA_MD5 = "29f284f36a0a510bcacf39ecf4c4d54f"


def _torch_cache_dir() -> Path:
    try:
        import torch
        root = Path(torch.hub.get_dir())
    except Exception:
        root = Path.home() / ".cache" / "torch" / "hub"
    path = root / "checkpoints"
    path.mkdir(parents=True, exist_ok=True)
    return path


def anime_lama_model_path() -> Path:
    return _torch_cache_dir() / "anime-manga-big-lama.pt"


def anime_lama_cached() -> bool:
    return anime_lama_model_path().is_file()


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_anime_lama() -> Path:
    target = anime_lama_model_path()
    if target.is_file() and _md5(target) == ANIME_LAMA_MD5:
        return target
    target.unlink(missing_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    print(f'Downloading anime/manga LaMa model to {target}')
    try:
        import torch
        torch.hub.download_url_to_file(ANIME_LAMA_URL, str(tmp), progress=True)
        if _md5(tmp) != ANIME_LAMA_MD5:
            raise LamaUnavailable("Downloaded anime/manga LaMa model failed its MD5 check.")
        tmp.replace(target)
    finally:
        tmp.unlink(missing_ok=True)
    return target


def lama_available() -> bool:
    try:
        import simple_lama_inpainting  # noqa: F401
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def _choose_device(device: Device):
    import torch
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise LamaUnavailable("CUDA was requested, but PyTorch cannot see a CUDA device.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _fit_backend_result(result: Image.Image, target_size: tuple[int, int]) -> Image.Image:
    """Return a backend image with exactly the original input dimensions.

    LaMa operates on dimensions padded to a model-friendly modulo. Some
    simple-lama-inpainting versions expose that padded result instead of
    cropping it back to the source size (for example 850 px -> 856 px).
    Its padding is on the bottom/right, so crop those edges when possible.

    The resize branch is only a defensive fallback for an unexpected backend
    that returns something *smaller* than the source image.
    """
    result = result.convert("RGB")
    if result.size == target_size:
        return result

    target_w, target_h = target_size
    result_w, result_h = result.size
    if result_w >= target_w and result_h >= target_h:
        return result.crop((0, 0, target_w, target_h))

    return result.resize(target_size, Image.Resampling.LANCZOS)


class Inpainter:
    def __init__(self, backend: Backend = "auto", device: Device = "auto") -> None:
        if backend == "auto":
            backend = "lama" if lama_available() else "opencv"
        self.backend = backend
        self.device = device
        self._lama = None
        self._anime_lama = None

        if self.backend == "lama" and not lama_available():
            raise LamaUnavailable(
                "LaMa backend is not installed. Run the setup script with GPU/CPU AI enabled, "
                "or install simple-lama-inpainting."
            )
        if self.backend == "anime-lama":
            try:
                import torch  # noqa: F401
            except Exception as exc:
                raise LamaUnavailable(
                    "Anime-LaMa requires PyTorch. Run the GPU/CPU AI setup first."
                ) from exc

    def _get_lama(self):
        if self._lama is None:
            from simple_lama_inpainting import SimpleLama
            self._lama = SimpleLama(device=_choose_device(self.device))
        return self._lama

    def _get_anime_lama(self):
        if self._anime_lama is None:
            import torch
            path = _download_anime_lama()
            self._anime_lama = torch.jit.load(str(path), map_location=_choose_device(self.device)).eval()
        return self._anime_lama

    def _run_anime_lama(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        import torch

        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        m = (np.asarray(mask.convert("L"), dtype=np.uint8) > 0).astype(np.float32)
        h, w = m.shape
        pad_h = (8 - h % 8) % 8
        pad_w = (8 - w % 8) % 8
        if pad_h or pad_w:
            rgb = np.pad(rgb, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            m = np.pad(m, ((0, pad_h), (0, pad_w)), mode="constant")

        image_t = torch.from_numpy(np.transpose(rgb, (2, 0, 1))).unsqueeze(0).to(_choose_device(self.device))
        mask_t = torch.from_numpy(m[None, None]).to(_choose_device(self.device))
        with torch.inference_mode():
            result = self._get_anime_lama()(image_t, mask_t)
        out = result[0].permute(1, 2, 0).detach().float().cpu().numpy()
        out = np.clip(out[:h, :w] * 255.0, 0, 255).astype(np.uint8)
        return Image.fromarray(out, "RGB")

    def run(self, image: Image.Image, mask: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        mask = mask.convert("L")
        if mask.size != image.size:
            # Masks are geometric data; nearest-neighbour avoids introducing
            # half-valued pixels around the removal region.
            mask = mask.resize(image.size, Image.Resampling.NEAREST)
        if not np.any(np.asarray(mask) > 0):
            return image.copy()

        if self.backend == "lama":
            result = self._get_lama()(image, mask)
            result = _fit_backend_result(result, image.size)

            # Defensive compositing: keep every unmasked pixel exactly identical
            # to the source image even if a backend modifies its context slightly.
            src = np.array(image)
            dst = np.array(result)
            m = np.array(mask) > 0
            dst[~m] = src[~m]
            return Image.fromarray(dst)

        if self.backend == "anime-lama":
            result = self._run_anime_lama(image, mask)
            src = np.array(image)
            dst = np.array(result)
            m = np.array(mask) > 0
            dst[~m] = src[~m]
            return Image.fromarray(dst)

        if self.backend == "opencv":
            bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            m = np.array(mask, dtype=np.uint8)
            # TELEA tends to behave nicely on thin text/line-art masks.
            out = cv2.inpaint(bgr, m, 3, cv2.INPAINT_TELEA)
            return Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))

        raise ValueError(f"Unknown backend: {self.backend}")
