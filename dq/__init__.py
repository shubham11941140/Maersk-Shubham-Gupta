"""Standalone data-quality checks that run directly against the raw CSV files.

This package has no dependency on the pipeline, the data store or the API, so
quality can be verified before anything is ingested::

    python -m dq --input ./data/raw --output ./artifacts/dq_report.json
"""
