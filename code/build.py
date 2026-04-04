#!/usr/bin/env python3
"""
Build script for the Glyphh Code model.

Unlike static models (toolrouter, churn), this model has no fixed exemplar
set. Each user compiles their own codebase via compile.py. This build script
packages the model's encoder config and MCP tool definitions into a .glyphh
artifact for deployment.

Usage:
    python build.py
"""

import sys
from pathlib import Path

from encoder import ENCODER_CONFIG


def build():
    """Package the code model for deployment."""
    try:
        from glyphh.model.package import GlyphhModel
    except ImportError:
        print("Error: glyphh SDK not installed. Run: pip install glyphh")
        sys.exit(1)

    model_dir = Path(__file__).parent

    model = GlyphhModel(
        name="code",
        version="0.1.0",
        encoder_config=ENCODER_CONFIG,
        glyphs=[],
        concepts=[],
        metadata={
            "author": "Glyphh AI",
            "description": "File-level codebase intelligence",
            "category": "search",
        },
    )

    errors = model.validate_completeness()
    if errors:
        print("Validation warnings:")
        for err in errors:
            print(f"  - {err}")

    output_path = model_dir / "model-code.glyphh"
    model.to_file(str(output_path))
    print(f"Built: {output_path} ({output_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    build()
