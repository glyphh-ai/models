#!/usr/bin/env python3
"""
Iris image ingestion — offline feature extraction pipeline.

Processes a directory of images → runs all feature extractors → writes
data/exemplars.jsonl in the format expected by entry_to_record().

Usage:
    python ingest.py /path/to/images
    python ingest.py /path/to/images --output data/exemplars.jsonl
    python ingest.py /path/to/images --workers 4
    python ingest.py /path/to/images --extractors color,lighting,composition

The output JSONL can then be loaded by the Glyphh runtime:
    glyphh model deploy iris/
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np

# Ensure the model directory is on the path (for imports like 'from encoder import ...')
_model_dir = os.path.dirname(os.path.abspath(__file__))
if _model_dir not in sys.path:
    sys.path.insert(0, _model_dir)

logger = logging.getLogger("iris.ingest")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}


def find_images(directory: str) -> List[Path]:
    """Recursively find all image files in a directory."""
    images = []
    for root, _, files in os.walk(directory):
        for f in sorted(files):
            if Path(f).suffix.lower() in IMAGE_EXTENSIONS:
                images.append(Path(root) / f)
    return images


def load_image(path: Path) -> np.ndarray:
    """Load an image as RGB numpy array."""
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    img = Image.open(path).convert("RGB")
    return np.array(img)


def extract_features(
    image_path: Path,
    extractor_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Extract features from a single image.

    Args:
        image_path: Path to image file
        extractor_names: Optional list of extractor names to run.
                         If None, runs all extractors.

    Returns:
        Dict with all extracted features, defaults filled in.
    """
    from extractors.base import ExtractorRegistry
    from extractors.face import FaceExtractor
    from extractors.pose import PoseExtractor
    from extractors.depth import DepthExtractor
    from extractors.ocr import OCRExtractor
    from extractors.objects import ObjectExtractor
    from extractors.color import ColorExtractor
    from extractors.lighting import LightingExtractor
    from extractors.composition import CompositionExtractor

    all_extractors = {
        "face": FaceExtractor,
        "pose": PoseExtractor,
        "depth": DepthExtractor,
        "ocr": OCRExtractor,
        "objects": ObjectExtractor,
        "color": ColorExtractor,
        "lighting": LightingExtractor,
        "composition": CompositionExtractor,
    }

    registry = ExtractorRegistry()
    names_to_use = extractor_names or list(all_extractors.keys())
    for name in names_to_use:
        if name in all_extractors:
            registry.register(name, all_extractors[name]())

    image = load_image(image_path)
    all_features = registry.extract_all(image)

    # Flatten extractor results into a single dict
    features = {}
    for _, extracted in all_features.items():
        features.update(extracted)

    # Fill defaults for any missing attributes
    from encoder import _fill_defaults
    _fill_defaults(features)

    return features


def process_image(
    image_path: str,
    base_dir: str,
    extractor_names: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Process a single image and return a JSONL entry.

    Args:
        image_path: Absolute path to image
        base_dir: Base directory for computing relative paths
        extractor_names: Optional extractor filter

    Returns:
        Dict ready for JSONL serialization, or None on failure.
    """
    path = Path(image_path)
    try:
        features = extract_features(path, extractor_names)

        # Generate stable image ID from relative path
        rel_path = str(path.relative_to(base_dir))
        image_id = rel_path.replace(os.sep, "/")

        # Convert numpy arrays to lists for JSON serialization
        serializable = {}
        for key, value in features.items():
            if isinstance(value, np.ndarray):
                serializable[key] = value.tolist()
            elif isinstance(value, (list, str, int, float, bool, type(None))):
                serializable[key] = value
            else:
                serializable[key] = str(value)

        return {
            "image_id": image_id,
            "features": serializable,
            "metadata": {
                "source_path": str(path),
                "file_size": path.stat().st_size,
                "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            },
        }
    except Exception as e:
        logger.error(f"Failed to process {image_path}: {e}")
        return None


def _process_image_wrapper(args):
    """Wrapper for multiprocessing — unpacks tuple args."""
    return process_image(*args)


def ingest(
    image_dir: str,
    output_path: str = "data/exemplars.jsonl",
    workers: int = 1,
    extractor_names: Optional[List[str]] = None,
    limit: Optional[int] = None,
):
    """
    Main ingestion pipeline.

    Args:
        image_dir: Directory containing images (searched recursively)
        output_path: Output JSONL file path
        workers: Number of parallel workers (1 = sequential)
        extractor_names: Optional list of extractor names to run
        limit: Optional max number of images to process
    """
    image_dir = os.path.abspath(image_dir)
    images = find_images(image_dir)

    if not images:
        logger.error(f"No images found in {image_dir}")
        return

    if limit:
        images = images[:limit]

    logger.info(f"Found {len(images)} images in {image_dir}")

    if extractor_names:
        logger.info(f"Running extractors: {', '.join(extractor_names)}")
    else:
        logger.info("Running all extractors")

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    processed = 0
    failed = 0
    start_time = time.time()

    with open(output_path, "w") as f:
        if workers > 1:
            args_list = [
                (str(img), image_dir, extractor_names) for img in images
            ]
            with ProcessPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(_process_image_wrapper, args): args[0]
                    for args in args_list
                }
                for future in as_completed(futures):
                    result = future.result()
                    if result:
                        f.write(json.dumps(result) + "\n")
                        processed += 1
                    else:
                        failed += 1
                    if (processed + failed) % 100 == 0:
                        logger.info(f"Progress: {processed + failed}/{len(images)}")
        else:
            for img in images:
                result = process_image(str(img), image_dir, extractor_names)
                if result:
                    f.write(json.dumps(result) + "\n")
                    processed += 1
                else:
                    failed += 1
                if (processed + failed) % 100 == 0:
                    logger.info(f"Progress: {processed + failed}/{len(images)}")

    elapsed = time.time() - start_time
    logger.info(
        f"Done: {processed} processed, {failed} failed, "
        f"{elapsed:.1f}s ({processed / max(elapsed, 0.001):.1f} img/s)"
    )
    logger.info(f"Output: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Iris offline image ingestion — extract features to exemplars.jsonl"
    )
    parser.add_argument(
        "image_dir",
        help="Directory containing images (searched recursively)",
    )
    parser.add_argument(
        "--output", "-o",
        default="data/exemplars.jsonl",
        help="Output JSONL path (default: data/exemplars.jsonl)",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )
    parser.add_argument(
        "--extractors", "-e",
        help="Comma-separated list of extractors to run (default: all). "
             "Options: face,pose,depth,ocr,objects,color,lighting,composition",
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        help="Max number of images to process",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    extractor_names = None
    if args.extractors:
        extractor_names = [e.strip() for e in args.extractors.split(",")]

    ingest(
        image_dir=args.image_dir,
        output_path=args.output,
        workers=args.workers,
        extractor_names=extractor_names,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
