"""Pure-data probe catalog fragments.

The run-report renderer is intentionally stdlib-only.  Keeping authored
catalog text in this dependency-free package lets it import the explanations
without importing numpy/torch-bearing probe executors.
"""
