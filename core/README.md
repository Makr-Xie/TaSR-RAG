This `core/` directory is the minimum local dependency closure for
`main_pipline/` from the original project.

Included:

- `main_pipline/`
- `matching/__init__.py`
- `matching/matching_mean.py`
- `matching/README.md`
- `utils/__init__.py`
- `utils/llm_client.py`
- `utils/text_tools.py`
- `config.py`
- `evaluate.py`
- `metrics.py`
- `build_faiss_type_index.py`

Why this works:

- `main_pipline/*.py` imports `config`, `utils`, and `matching.matching_mean`.
- The copied directory layout preserves those imports without changing code.
- The `sys.path` handling inside the original scripts still resolves against
  `core/`.

What is still intentionally missing:

- `type_faiss/`
- datasets
- result files
- serving scripts
- ablation and analysis folders

Recommended next cleanup steps:

1. Rename `main_pipline` to `pipeline` once behavior is frozen.
2. Replace ad hoc `sys.path` manipulation with package-relative imports.
3. Move hard-coded model settings out of `config.py`.
4. Convert script variants into parameterized runners instead of separate files.
