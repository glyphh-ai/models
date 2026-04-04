"""Pose feature extractor — body keypoints and head orientation."""

import numpy as np
from typing import Any, Dict
from extractors.base import FeatureExtractor


class PoseExtractor(FeatureExtractor):
    """Extract body pose (33 keypoints x 3) and head pose (pitch/yaw/roll)."""

    def __init__(self):
        self._detector = None

    def is_available(self) -> bool:
        try:
            import mediapipe as mp  # noqa: F401
            # Check for new Tasks API (mediapipe >= 0.10.8)
            from mediapipe.tasks.python import vision  # noqa: F401
            return True
        except (ImportError, AttributeError):
            return False

    def _load_model(self):
        if self._detector is None:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision

            options = vision.PoseLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=self._get_model_path(),
                ),
                running_mode=vision.RunningMode.IMAGE,
                num_poses=1,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
            )
            self._detector = vision.PoseLandmarker.create_from_options(options)

    def _get_model_path(self) -> str:
        """Download pose landmarker model if needed, return path."""
        import os
        import urllib.request

        cache_dir = os.path.expanduser("~/.cache/mediapipe/models")
        os.makedirs(cache_dir, exist_ok=True)
        model_path = os.path.join(cache_dir, "pose_landmarker_lite.task")

        if not os.path.exists(model_path):
            url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
            urllib.request.urlretrieve(url, model_path)

        return model_path

    def extract(self, image: np.ndarray) -> Dict[str, Any]:
        self._load_model()
        import mediapipe as mp

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
        result = self._detector.detect(mp_image)

        if not result.pose_landmarks or len(result.pose_landmarks) == 0:
            return self.defaults()

        landmarks = result.pose_landmarks[0]

        # 33 keypoints × 3 (x, y, z) = 99 floats
        body_pose = []
        for lm in landmarks:
            body_pose.extend([lm.x, lm.y, lm.z])

        # Approximate head pose from nose/ear/eye landmarks
        nose = landmarks[0]
        left_ear = landmarks[7]
        right_ear = landmarks[8]

        yaw = (left_ear.x - right_ear.x)
        pitch = nose.y - (left_ear.y + right_ear.y) / 2
        roll = left_ear.y - right_ear.y

        return {
            "body_pose": body_pose,
            "head_pose": [pitch, yaw, roll],
        }

    def defaults(self) -> Dict[str, Any]:
        return {
            "body_pose": [0.0] * 99,
            "head_pose": [0.0, 0.0, 0.0],
        }
