"""
Manifest utilities for dataset indexing and split management.

This module defines helpers to create, validate, load, and save dataset
manifest files used across the pipeline (baseline, rerank, reward-filtering,
and post-training).

In this project, a manifest is the canonical index of samples (e.g., complexes)
and their resolved file paths/metadata. Using manifests ensures:
- reproducible train/val/test splits
- consistent sample selection across runs
- simple config-driven execution via manifest_path entries
"""