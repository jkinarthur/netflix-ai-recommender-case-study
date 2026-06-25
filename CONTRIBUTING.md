# Contributing Guide

Thanks for your interest in contributing.

## Development Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

3. Run the pipeline once:

```bash
python run_demo.py
```

4. Launch the app:

```bash
streamlit run app.py
```

## Contribution Workflow

1. Fork the repository.
2. Create a feature branch: `feature/<short-name>`.
3. Keep changes focused and atomic.
4. Add or update documentation when behavior changes.
5. Run local checks before opening a PR.

## Pull Request Checklist

- [ ] Code builds and runs locally.
- [ ] No generated `data/` or `outputs/` artifacts committed.
- [ ] README/docs updated where applicable.
- [ ] PR description clearly states intent and impact.

## Style Guidance

- Prefer clear, explicit naming.
- Keep functions small and single-purpose.
- Add concise comments only when logic is non-obvious.
- Preserve reproducibility for metrics and charts.
