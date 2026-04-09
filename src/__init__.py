"""Final_version package for OCR training and inference."""

# Setup environment for headless OpenCV BEFORE any imports
import os
os.environ['DISPLAY'] = ''
os.environ['LIBGL_ALWAYS_INDIRECT'] = '1'

from .config import ProjectPaths, TrainingConfig, get_project_paths, get_training_config
