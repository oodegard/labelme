from __future__ import annotations

import collections
import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from typing import Final

import numpy as np
from loguru import logger
from numpy.typing import NDArray
from segment_anything import SamPredictor
from segment_anything import sam_model_registry

import xxhash


_MODEL_URLS: dict[str, str] = {
    "vit_l_lm": "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/idealistic-rat/1.2/files/vit_l.pt",
    "vit_b_lm": "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/diplomatic-bug/1.2/files/vit_b.pt",
    "vit_b_em_organelles": "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/noisy-ox/1.2/files/vit_b.pt",
}

_MODEL_HASHES: dict[str, str] = {
    "vit_l_lm": "xxh128:017f20677997d628426dec80a8018f9d",
    "vit_b_lm": "xxh128:fe9252a29f3f4ea53c15a06de471e186",
    "vit_b_em_organelles": "xxh128:f3bf2ed83d691456bae2c3f9a05fb438",
}

_MODEL_BACKBONES: dict[str, str] = {
    "vit_l_lm": "vit_l",
    "vit_b_lm": "vit_b",
    "vit_b_em_organelles": "vit_b",
}

_HASH_CHUNK_SIZE: Final[int] = 8192


@dataclass
class _MicroSamAnnotation:
    mask: NDArray[np.bool_]
    bounding_box: None = None


@dataclass
class _MicroSamGenerateResponse:
    annotations: list[_MicroSamAnnotation]


class MicroSamSession:
    _model_name: str
    _predictor: SamPredictor | None
    _loaded_image_id: str | None
    _embedding_cache: collections.deque[str]

    def __init__(
        self,
        model_name: str = "microsam:vit_b_lm",
        embedding_cache_size: int = 3,
    ) -> None:
        logger.debug("Initializing MicroSamSession with model_name={!r}", model_name)
        self._model_name = model_name
        self._predictor = None
        self._loaded_image_id = None
        self._embedding_cache = collections.deque(maxlen=embedding_cache_size)
        logger.debug("Initialized MicroSamSession with model_name={!r}", model_name)

    @property
    def model_name(self) -> str:
        return self._model_name

    def run(
        self,
        image: NDArray[np.uint8],
        image_id: str,
        points: NDArray[np.floating] | None = None,
        point_labels: NDArray[np.intp] | None = None,
        texts: list[str] | None = None,
    ) -> _MicroSamGenerateResponse:
        if texts is not None:
            raise NotImplementedError("micro-sam backend does not support text prompts")
        if points is None or point_labels is None:
            raise ValueError("points and point_labels are required for micro-sam")

        predictor = self._get_or_load_predictor()
        self._set_image_if_needed(
            predictor=predictor,
            image=image,
            image_id=image_id,
        )

        masks, _scores, _logits = predictor.predict(
            point_coords=np.asarray(points, dtype=np.float64),
            point_labels=np.asarray(point_labels, dtype=np.uint8),
            multimask_output=False,
        )

        if isinstance(masks, np.ndarray) and masks.ndim == 3:
            mask = masks[0]
        else:
            mask = np.asarray(masks)
        mask = mask.astype(bool)

        return _MicroSamGenerateResponse(
            annotations=[_MicroSamAnnotation(mask=mask)],
        )

    def _set_image_if_needed(
        self,
        predictor: SamPredictor,
        image: NDArray[np.uint8],
        image_id: str,
    ) -> None:
        if self._loaded_image_id == image_id:
            return

        image_rgb = image
        if image_rgb.ndim == 2:
            image_rgb = np.stack([image_rgb, image_rgb, image_rgb], axis=-1)
        elif image_rgb.ndim == 3 and image_rgb.shape[-1] == 1:
            image_rgb = np.repeat(image_rgb, 3, axis=-1)
        elif image_rgb.ndim != 3 or image_rgb.shape[-1] not in [3, 4]:
            raise ValueError(f"Unsupported image shape for MicroSAM: {image_rgb.shape}")

        if image_rgb.shape[-1] == 4:
            image_rgb = image_rgb[..., :3]

        if image_rgb.dtype != np.uint8:
            image_rgb = np.clip(image_rgb, 0, 255).astype(np.uint8)

        logger.debug("Setting image for micro-sam predictor: image_id={!r}", image_id)
        predictor.set_image(image_rgb)
        self._loaded_image_id = image_id
        self._embedding_cache.append(image_id)

    def _get_or_load_predictor(self) -> SamPredictor:
        return self._get_or_load_predictor_with_progress(progress_callback=None)

    def _get_or_load_predictor_with_progress(
        self,
        progress_callback: Callable[[int, int], None] | None,
    ) -> SamPredictor:
        if self._predictor is None:
            model_type = self._get_model_type()
            if model_type not in _MODEL_URLS:
                raise ValueError(
                    "Unsupported micro-sam model type: "
                    f"{model_type!r}. Supported: {sorted(_MODEL_URLS)}"
                )

            checkpoint = self._ensure_checkpoint(
                model_type=model_type,
                progress_callback=progress_callback,
            )
            backbone = _MODEL_BACKBONES[model_type]
            logger.debug(
                "Loading micro-sam checkpoint with model_type={!r}, backbone={!r}",
                model_type,
                backbone,
            )
            model = sam_model_registry[backbone](checkpoint=checkpoint)
            self._predictor = SamPredictor(model)
            logger.debug("Loaded micro-sam checkpoint for model_type={!r}", model_type)
        return self._predictor

    def _ensure_checkpoint(
        self,
        model_type: str,
        progress_callback: Callable[[int, int], None] | None,
    ) -> str:
        cache_dir = self._get_cache_dir() / "models"
        cache_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = cache_dir / f"{model_type}.pt"
        expected_hash = _MODEL_HASHES[model_type]

        if checkpoint_path.exists() and checkpoint_path.stat().st_size > 0:
            actual_hash = self._compute_hash(checkpoint_path)
            if actual_hash == expected_hash:
                return str(checkpoint_path)
            logger.warning(
                "Micro-sam checkpoint hash mismatch for {!r}: expected {!r}, got {!r}. Redownloading.",
                checkpoint_path,
                expected_hash,
                actual_hash,
            )

        url = _MODEL_URLS[model_type]
        logger.info("Downloading micro-sam checkpoint: {!r} -> {!r}", url, checkpoint_path)
        tmp_path = checkpoint_path.with_suffix(".pt.tmp")

        def reporthook(block_count: int, block_size: int, total_size: int) -> None:
            if progress_callback is None:
                return
            downloaded = block_count * block_size
            progress_callback(downloaded, total_size)

        try:
            urllib.request.urlretrieve(url, tmp_path, reporthook=reporthook)  # noqa: S310
            os.replace(tmp_path, checkpoint_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

        actual_hash = self._compute_hash(checkpoint_path)
        if actual_hash != expected_hash:
            raise RuntimeError(
                "Downloaded micro-sam checkpoint has invalid hash: "
                f"expected {expected_hash}, got {actual_hash}"
            )
        return str(checkpoint_path)

    def _get_cache_dir(self) -> Path:
        cachedir_env = os.environ.get("MICROSAM_CACHEDIR")
        if cachedir_env:
            return Path(cachedir_env)

        if os.name == "nt":
            local_app_data = os.environ.get("LOCALAPPDATA")
            if local_app_data:
                return Path(local_app_data) / "micro_sam" / "Cache"
            return Path.home() / "AppData" / "Local" / "micro_sam" / "Cache"

        xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
        if xdg_cache_home:
            return Path(xdg_cache_home) / "micro_sam"
        return Path.home() / ".cache" / "micro_sam"

    def _compute_hash(self, path: Path) -> str:
        hash_obj = xxhash.xxh128()
        with path.open("rb") as file_obj:
            chunk = file_obj.read(_HASH_CHUNK_SIZE)
            while chunk:
                hash_obj.update(chunk)
                chunk = file_obj.read(_HASH_CHUNK_SIZE)
        return f"xxh128:{hash_obj.hexdigest()}"

    def _get_model_type(self) -> str:
        if not self._model_name.startswith("microsam:"):
            raise ValueError(
                f"micro-sam model_name must start with 'microsam:', got {self._model_name!r}"
            )
        return self._model_name.split(":", 1)[1]