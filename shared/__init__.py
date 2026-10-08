"""Code shared by the DQ checks, the pipeline, the ML model and the API.

Keeping domain rules (what counts as "on time", which status values are
canonical, etc.) in one place stops the four components from drifting apart.
"""
