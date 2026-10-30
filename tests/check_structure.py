"""Lightweight repository checks with no external ML dependencies."""
from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    'video_streaming_detector_simple_optimized_v2.py',
    'video_streaming_detector_fast_v2.py',
    'compare_pipeline_methods_batched.py',
    'utils/__init__.py',
    'utils/models.py',
    'utils/load_efficientad_adaptive.py',
    'models_adaptive.py',
    'models_thin.py',
    'models_fixed.py',
    'README.md',
    'requirements.txt',
]
for path in REQUIRED:
    assert (ROOT / path).is_file(), f'Missing required file: {path}'

sources = list(ROOT.rglob('*.py'))
assert sources, 'No Python sources found'
for path in sources:
    if any(p in {'.git', '.venv', 'venv'} for p in path.parts):
        continue
    ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
print(f'Validated repository structure and parsed {len(sources)} Python files.')
