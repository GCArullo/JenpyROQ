# Building the documentation locally

Install the documentation dependencies from the repository root:

```bash
python -m pip install .[docs]
```

Build the HTML documentation:

```bash
sphinx-build -b html docs docs/_build/html
```
