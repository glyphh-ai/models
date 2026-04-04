#!/usr/bin/env python3
"""
Package a model directory into a .glyphh file for release.

Usage:
    python scripts/publish.py <model_dir> [--version <version>]

The .glyphh file is a zip archive containing everything needed to deploy
the model. Includes all Python files, config, data, hooks, and tests.
Excludes build artifacts, caches, and dev files.
"""

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

import yaml

# Files/dirs to always exclude from the package
EXCLUDE_PATTERNS = {
    "__pycache__",
    ".pytest_cache",
    ".venv",
    ".git",
    ".gitignore",
    ".DS_Store",
    "*.pyc",
    "*.pyo",
    "*.egg-info",
    "*.glyphh",
    "dist",
    "build",
    "node_modules",
    "README.md",
    "LICENSE",
    "requirements.txt",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
}


def _should_exclude(path: Path) -> bool:
    """Check if a path should be excluded from packaging."""
    for part in path.parts:
        if part in EXCLUDE_PATTERNS:
            return True
        if part.startswith(".") and part not in (".glyphh",):
            return True
    if path.suffix in (".pyc", ".pyo"):
        return True
    if path.name.endswith(".egg-info"):
        return True
    return False


def build_model(model_dir: Path) -> bool:
    """Run build.py if it exists. Returns True on success."""
    build_script = model_dir / "build.py"
    if not build_script.exists():
        return True

    print(f"  Running build.py...")
    result = subprocess.run(
        [sys.executable, str(build_script)],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=str(model_dir),
        env={**__import__("os").environ, "PYTHONPATH": str(model_dir)},
    )
    if result.returncode != 0:
        print(f"  Build failed: {result.stderr.strip()[:300]}", file=sys.stderr)
        return False
    if result.stdout.strip():
        print(f"  {result.stdout.strip()}")
    return True


def package_model(model_dir: Path, version: str | None = None) -> Path:
    """Package a model directory into a .glyphh zip file."""
    manifest_path = model_dir / "manifest.yaml"
    if not manifest_path.exists():
        print(f"Error: {manifest_path} not found", file=sys.stderr)
        sys.exit(1)

    manifest = yaml.safe_load(manifest_path.read_text()) or {}
    model_id = manifest.get("model_id", model_dir.name)

    if version:
        manifest["version"] = version

    out_file = Path(f"{model_id}.glyphh")

    with zipfile.ZipFile(out_file, "w", zipfile.ZIP_DEFLATED) as zf:
        # Write manifest (possibly with updated version)
        zf.writestr(
            "manifest.yaml",
            yaml.dump(manifest, default_flow_style=False, sort_keys=False),
        )

        # Walk the model directory and include everything not excluded
        for item in sorted(model_dir.rglob("*")):
            if not item.is_file():
                continue
            rel = item.relative_to(model_dir)
            if rel.parts[0] == "manifest.yaml":
                continue  # already written with version override
            if _should_exclude(rel):
                continue
            zf.write(item, str(rel))

    size_kb = out_file.stat().st_size / 1024
    print(f"Packaged {model_id} v{manifest.get('version', '?')} -> {out_file} ({size_kb:.1f} KB)")
    return out_file


def main():
    parser = argparse.ArgumentParser(description="Package a Glyphh model")
    parser.add_argument("model_dir", type=Path, help="Path to model directory")
    parser.add_argument("--version", "-v", help="Override version in manifest")
    parser.add_argument("--skip-build", action="store_true", help="Skip build.py")
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()

    if not args.skip_build:
        if not build_model(model_dir):
            sys.exit(1)

    package_model(model_dir, args.version)


if __name__ == "__main__":
    main()
