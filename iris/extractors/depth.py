"""Depth feature extractor — monocular depth estimation."""

import numpy as np
from typing import Any, Dict
from extractors.base import FeatureExtractor


class DepthExtractor(FeatureExtractor):
    """Extract depth map, pooled to 8x8 grid (64-dim float vector)."""

    def __init__(self, grid_size=(8, 8)):
        self._model = None
        self._transform = None
        self.grid_size = grid_size

    def is_available(self) -> bool:
        try:
            import torch  # noqa: F401
            import timm  # noqa: F401
            return True
        except ImportError:
            return False

    def _load_model(self):
        if self._model is None:
            import torch
            self._model = torch.hub.load(
                "intel-isl/MiDaS", "MiDaS_small", trust_repo=True
            )
            self._model.eval()
            midas_transforms = torch.hub.load(
                "intel-isl/MiDaS", "transforms", trust_repo=True
            )
            self._transform = midas_transforms.small_transform

    def extract(self, image: np.ndarray) -> Dict[str, Any]:
        self._load_model()
        import torch

        input_batch = self._transform(image)
        with torch.no_grad():
            prediction = self._model(input_batch)
            depth_map = prediction.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        d_min, d_max = depth_map.min(), depth_map.max()
        if d_max - d_min > 0:
            depth_map = (depth_map - d_min) / (d_max - d_min)
        else:
            depth_map = np.zeros_like(depth_map)

        # Pool to grid
        pooled = self._pool_to_grid(depth_map)

        return {
            "depth_map": pooled.flatten().tolist(),
            "depth_range": float(d_max - d_min),
        }

    def _pool_to_grid(self, depth_map: np.ndarray) -> np.ndarray:
        """Average-pool depth map to fixed grid."""
        h, w = depth_map.shape
        gh, gw = self.grid_size
        result = np.zeros((gh, gw), dtype=np.float32)
        bh, bw = h / gh, w / gw
        for i in range(gh):
            r0, r1 = int(round(i * bh)), int(round((i + 1) * bh))
            for j in range(gw):
                c0, c1 = int(round(j * bw)), int(round((j + 1) * bw))
                block = depth_map[r0:r1, c0:c1]
                if block.size > 0:
                    result[i, j] = block.mean()
        return result

    def defaults(self) -> Dict[str, Any]:
        return {
            "depth_map": [0.0] * (self.grid_size[0] * self.grid_size[1]),
            "depth_range": 0.0,
        }
